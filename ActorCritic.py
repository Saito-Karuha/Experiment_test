from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import torch

class CriticNet(torch.nn.Module):
    '''
    这里的Critic-Net从Sparse Signals中训练而来,
    base_model选用MetaMath-Mistral-7b,所以下面的代码全部都服务于这个模型，其他模型不可用；
    cache_dir表示模型路径'''
    def __init__(self, base_model:str, cache_dir:str, device:str="cpu"): # 更改模型
        super().__init__()
        self.device = torch.device(device)
        self.base_model = AutoModel.from_pretrained(cache_dir, trust_remote_code=True).half().to(self.device) #######
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True, use_fast=False) #########
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.pad_token_id = self.tokenizer.pad_token_id # [PAD] 32000
        self.eos_token_id = self.tokenizer.eos_token_id # <s> 1
        self.bos_token_id = self.tokenizer.bos_token_id # </s> 2
        # 输出层，将最后最后一个token的表示用mlp转化为对应的PRM分数
        # 这里的输入数据格式必须符合"Step 1:....ки\n Step 2:...ки\n"以此类推，注意为满足这样的格式，actorNet可能需要微调
        self.v_head_mlp1 = torch.nn.Linear(self.hidden_size, 1024, bias=False).to(self.device)
        self.v_head_mlp2 = torch.nn.Linear(1024, 512, bias=False).to(self.device)
        self.v_head_mlp3 = torch.nn.Linear(512, 1, bias=False).to(self.device)
        self.relu = torch.nn.ReLU()

    def save_model(self, save_path: str):
        """保存模型参数"""
        state_dict = {
            'v_head_mlp1': self.v_head_mlp1.state_dict(),
            'v_head_mlp2': self.v_head_mlp2.state_dict(),
            'v_head_mlp3': self.v_head_mlp3.state_dict()
        }
        torch.save(state_dict, save_path)

    def load_model(self, load_path: str):
        """加载模型参数"""
        state_dict = torch.load(load_path, map_location=self.device)
        self.v_head_mlp1.load_state_dict(state_dict['v_head_mlp1'])
        self.v_head_mlp2.load_state_dict(state_dict['v_head_mlp2'])
        self.v_head_mlp3.load_state_dict(state_dict['v_head_mlp3'])

    def get_trainable_parameters(self):
        return [
            {'params': self.v_head_mlp1.parameters()},
            {'params': self.v_head_mlp2.parameters()},
            {'params': self.v_head_mlp3.parameters()}
        ]

    def forward(self, text:str):
        '''这里的input_ids和attention_mask都必须是torch.tensor,并且shape均为(1,n);
        现在这里只接收单一的句子输入
        新更正: 现在把 forward改为接收字符串直接输出, text是 question+"\nSolution:\n"+pre_reasoning'''
        with torch.no_grad():
            tokenized_sentence = self.tokenizer(text, return_tensors='pt')
            tokenized_sentence = {k: v.to(self.device) for k, v in tokenized_sentence.items()}  # 添加这行
            input_ids = tokenized_sentence['input_ids']
            attention_mask = tokenized_sentence['attention_mask']
            baseModel_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            baseModel_outputs = baseModel_outputs.last_hidden_state[:,-1,:].float() # shape:(句子数,self.hidden_size)
        # 上面输出的 dtype=torch.float32
        x = self.relu(self.v_head_mlp1(baseModel_outputs))
        x = self.relu(self.v_head_mlp2(x))
        x = self.v_head_mlp3(x) # shape:(句子数,1)
        x = torch.nn.functional.sigmoid(x)
        return x


class ActionNet(torch.nn.Module):
    '''这里选用模型 peiyi9979/mistral-7b-sft, 输出严格遵循确定的格式, 格式如下:
    Step 1: <content> ки\nStep 2 ,,,以此类推; 注意<content>中可以包含\n, 我们真正需要的 step-tag是 'ки\n'
    (但是还是建议加上 verifier防止训练时炸掉)'''

    def __init__(self, base_model: str, cache_dir: str, chat_template: str, device:str="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.base_model = AutoModelForCausalLM.from_pretrained(cache_dir, pad_token_id=2).half().to(self.device)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,
            lora_dropout=0.1  # Dropout 比例
        )
        self.base_model = get_peft_model(self.base_model, config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir, chat_template=chat_template, pad_token_id=2)
        self.chat_template = chat_template
        self.special_tag = 'ки'
        self.special_tag_id = self.tokenizer.encode(self.special_tag)[1]  # 12902

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def save_model(self, save_path: str):
        """保存 LoRA 模型参数"""
        self.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        """加载 LoRA 模型参数"""
        self.base_model.load_adapter(load_path)

    def forward(self, question: str = None, pre_reasoning: str = None, num_gen: int = 1, if_return_logits: bool = False):
        '''前向推理函数，生成下一步的推理
        (question, pre_reasoning)都是字符串, 即自然语言形式表述的问题以及前述推理步骤, num_gen表示生成数
        返回值, 'response_ids':下一步推理的 id_tensor:[tensor1,tensor2,...], 'nl_response':[str1,str2,...]
        表示下一步解决方案的自然语言格式, 'actual_numGen':表示实际生成的方案数(不一定等于num_gen)'''
        assert question != None
        if pre_reasoning == None:
                 inputs =  "system: Please reason step by step, and put your final answer within \\boxed{}.\n" + question + "\nSolution:\n"
        else:
                 inputs =  "system: Please reason step by step, and put your final answer within \\boxed{}.\n" + question + "\nSolution:\n" + pre_reasoning
        model_inputs = self.tokenizer([inputs], return_tensors='pt')
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}  # 添加这行
        generated_ids = self.base_model.generate(
            **model_inputs,
            temperature=0.5,
            do_sample=True,
            num_return_sequences=num_gen,
            max_new_tokens=256,
            return_dict_in_generate=True,
            eos_token_id=[self.tokenizer.eos_token_id, self.special_tag_id]
        )  # 会生成一个列表, 大小是[num_gen,文本长度+Pad]
        generated_ids = generated_ids.sequences
        generated_ids.detach()

        next_steps = []
        nl_response = []
        log_probs = []
        origin_logits = []
        input_length = len(model_inputs["input_ids"][0])

        for i in range(len(generated_ids)):
            # 获取生成的序列（不包含输入序列）
            generated_sequence = generated_ids[i][input_length:]

            # 找到第一个pad token或eos token的位置, 确定实际序列的结束位置
            pad_pos = torch.where(generated_sequence == self.tokenizer.pad_token_id)[0]
            eos_pos = torch.where(generated_sequence == self.tokenizer.eos_token_id)[0]
            end_pos = None
            if len(pad_pos) > 0:
                end_pos = pad_pos[0] if end_pos is None else min(end_pos, pad_pos[0])
            if len(eos_pos) > 0:
                end_pos = eos_pos[0] if end_pos is None else min(end_pos, eos_pos[0])

                # 如果没有找到pad或eos token，使用整个序列
            if end_pos is None:
                end_pos = len(generated_sequence)

                # 截取实际生成的序列（不包含pad和eos token）
            actual_sequence = generated_sequence[:end_pos]
            next_steps.append(actual_sequence)
            nl_text = self.tokenizer.decode(actual_sequence)
            nl_response.append(nl_text)

            if if_return_logits:
                # 计算对数概率
                logits = self.base_model(input_ids=generated_ids[i].unsqueeze(0),
                                         attention_mask=torch.ones_like(generated_ids[i].unsqueeze(0)),
                                         return_dict=True).logits[0][
                         input_length - 1:input_length + len(actual_sequence) - 1, :]

                probs = torch.nn.functional.softmax(logits, dim=-1)
                logits_word = torch.gather(probs, dim=-1, index=actual_sequence.unsqueeze(-1)).squeeze(-1)
                origin_logits.append(logits_word)
                token_log_probs = torch.log(logits_word)
                avg_log_prob = token_log_probs.mean()
                log_probs.append(avg_log_prob)

        return {
            'response_ids': next_steps,
            'nl_response': nl_response,
            'actual_numGen': len(next_steps),
            'correspond_logits': origin_logits,
            'avg_log_probs': log_probs
        }

    def get_log_prob(self, question:str=None, pre_reasoning:str=None, action_took:str=None):
        '''这里主要是去求在当前参数下生成 action_took的置信度(或许我们需要把原来的 prompt也加进去?)'''
        log_probs = []
        origin_logits = []

        prefix = "system: Please reason step by step, and put your final answer within \\boxed{}.\n" + question + "\nSolution:\n" + pre_reasoning
        text = prefix + action_took
        prefix_inputs = self.tokenizer([prefix], return_tensors='pt')
        prefix_inputs = {k: v.to(self.device) for k, v in prefix_inputs.items()}
        prefix_length = len(prefix_inputs['input_ids'][0])
        inputs = self.tokenizer([text], return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.base_model(input_ids=inputs['input_ids'], attention_mask=torch.ones_like(inputs['input_ids']).to(self.device),
                                 return_dict=True).logits[0][prefix_length-1:,:]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logits_word = torch.gather(probs, dim=-1, index=inputs['input_ids'][0][prefix_length:].unsqueeze(-1)).squeeze(-1)
        origin_logits.append(logits_word)
        token_log_probs = torch.log(logits_word)
        avg_log_prob = token_log_probs.mean()
        log_probs.append(avg_log_prob)

        return {'correspond_logits': origin_logits, 'avg_log_probs': log_probs}