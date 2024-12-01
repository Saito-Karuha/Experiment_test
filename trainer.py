# 这里开始搭建整个训练流程
import os
import json
import random
import re
from collections import deque
import random
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from ActorCritic import CriticNet, ActionNet
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from preprocess import *
from replaybuffer import *
import torch
import warnings

actor_config = {"base_model": r"peiyi9979/mistral-7b-sft", "cache_dir": "./models--peiyi9979--mistral-7b-sft",
                "chat_template": r"{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '<s> ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"}
critic_config = {'base_model': r"meta-math/MetaMath-Mistral-7B",
                 "cache_dir": "./models--meta-math--MetaMath-Mistral-7B"}


# 对显存进行监控
def print_gpu_memory():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f}MB")


# 设置目标网络(actor, critic)的更新函数 (这里使用 Soft Update来进行)
def soft_update_actor(actor, actor_target, tau=0.005):
    '''软更新 targetNet lora参数部分'''
    actor_lora_state_dict = actor.base_model.state_dict()
    target_lora_state_dict = actor_target.base_model.state_dict()

    new_target_dict = {}
    for key in actor_lora_state_dict:
        if 'lora_' in key:
            new_target_dict[key] = (1 - tau) * target_lora_state_dict[key] + tau * actor_lora_state_dict[key]
    # 更新目标网络的权重
    actor_target.base_model.load_state_dict(new_target_dict, strict=False)


def soft_update_critic(critic, critic_target, tau=0.005):
    with torch.no_grad():
        # 获取两个模型的状态字典
        critic_state_dict = critic.state_dict()
        target_state_dict = critic_target.state_dict()

        # 只更新 MLP 层的参数
        for name, param in critic_state_dict.items():
            if 'v_head_mlp' in name:
                # 软更新公式：target_params = (1 - tau) * target_params + tau * source_params
                target_state_dict[name].copy_(
                    target_state_dict[name] * (1 - tau) + param * tau
                )


# 设置actor net的优化器
def create_lora_optimizer(model, learning_rate=1e-4, weight_decay=0.01, betas=(0.9, 0.999)):
    # 获取所有需要优化的LoRA参数
    lora_params = []
    for name, param in model.base_model.named_parameters():
        if 'lora_' in name and param.requires_grad:  # 只选择需要梯度的LoRA参数
            lora_params.append(param)

    # 创建优化器
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )

    return optimizer


def train(actor_config: dict, critic_config: dict, epoch: int, dataset_dir: str, max_reason_length: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 首先做一个initialize(包括 target net和普通的更新目标)
    critic = CriticNet(base_model=critic_config['base_model'], cache_dir=critic_config["cache_dir"], device="cuda")
    actor = ActionNet(base_model=actor_config['base_model'], cache_dir=actor_config['cache_dir'],
                      chat_template=actor_config['chat_template'], device="cuda")
    critic_target = CriticNet(base_model=critic_config['base_model'], cache_dir=critic_config["cache_dir"],
                              device="cuda")
    actor_target = ActionNet(base_model=actor_config['base_model'], cache_dir=actor_config['cache_dir'],
                             chat_template=actor_config['chat_template'], device="cuda")

    # 对基础模型启用梯度检查点
    critic.base_model.gradient_checkpointing_enable()
    actor.base_model.gradient_checkpointing_enable()

    # 对目标网络的基础模型也启用梯度检查点
    critic_target.base_model.gradient_checkpointing_enable()
    actor_target.base_model.gradient_checkpointing_enable()

    # 对 CriticNet_target的权重转移以及 Optimizer的设置
    with torch.no_grad():
        for name, param in critic.state_dict().items():
            if 'v_head_mlp' in name:
                critic_target.state_dict()[name].copy_(param)

    optimizer_critic = torch.optim.AdamW(critic.get_trainable_parameters(), lr=1e-5)

    # 对 Actornet_target 的权重转移以及 Optimizer的设置

    with torch.no_grad():
        lora_state_dict = {k: v for k, v in actor.base_model.state_dict().items() if "lora_" in k}
        actor_target.base_model.load_state_dict(lora_state_dict, strict=False)

    optimizer_actor = create_lora_optimizer(actor)
    # 设置环境
    env = Env(dataset_dir)
    # 设置经验回放池
    buffer = ReplayBuffer(5000)
    minimal_size = 50
    batch_size = 10
    eps = 0.2

    # 为epoch循环添加进度条
    epoch_pbar = tqdm(range(epoch), desc='Training Progress')
    total_critic_loss = 0
    total_actor_loss = 0

    gamma = 0.98  # discount factor

    # 正式训练流程
    for e in epoch_pbar:
        x = env.sample(split="train")  # 每个epoch都从训练集中随机选择一个问题
        question = x[0]['problem']
        state = f""  # 实际上这个state是我们的 pre_reasoning的实现, But!在使用Critic Net的时候需要结合 problem
        ground_truth = x[0]['ground_truth']
        done = False

        # 数据收集
        with torch.no_grad():
            for t in range(1, max_reason_length + 1):
                # 单步推理
                tep = actor.forward(question=question, pre_reasoning=state, num_gen=1, if_return_logits=False)
                action = tep['nl_response'][0]
                reward, next_state, done = env.step(state, action, ground_truth)
                buffer.add(state, action, reward, next_state, question)
                state = next_state
                if done:
                    break

        ##########
        # 进行经验回放进行训练
        if buffer.size() >= minimal_size:
            replay_set = buffer.sample(batch_size)  # 结构大致为[(s1,a,r,s2,q),(...)...]
            L, J = torch.tensor([0], dtype=torch.float32, device=device), torch.tensor([0], dtype=torch.float32,
                                                                                       device=device)  # 损失函数
            for i in range(batch_size):
                s1, a, r, s2, q = replay_set[i]
                with torch.no_grad():
                    ans = actor_target.forward(question=q, pre_reasoning=s2, num_gen=4)
                    ans = ans['nl_response']  # 这里固定的在 4个 action之中去找最大值, ans=[str1,str2,...]
                    value_batch = torch.zeros((len(ans),), dtype=torch.float32).to(device)  # [v1,v2,...]
                    for j in range(len(ans)):
                        value_batch[j] = (critic_target.forward(q + "\nSolution:\n" + s2 + ans[j])[0][0])
                    value_batch.detach_()  # 分离计算图
                # 得到 criticNet的损失函数
                v1 = critic.forward(q + "\nSolution:\n" + s1 + a)[0][0]
                L += (v1 - (gamma * value_batch.max().item() + r)) ** 2

                # 下面处理 actorNet的部分
                random_index = torch.randint(0, len(ans), (1,)).item()
                advantage = (r + gamma*value_batch[random_index]) - v1.detach().item()
                # 这里是用置信度作为概率的替代吗？？？
                old_log_p = actor_target.get_log_prob(q, s1, a)['avg_log_probs'][0]
                old_log_p.detach_()  # 截断梯度
                new_log_p = actor.get_log_prob(q, s1, a)['avg_log_probs'][0]
                ratio = torch.exp(new_log_p - old_log_p)
                ######## 下面这个式子需要 double check一下
                J += torch.min(ratio * advantage, torch.clip(ratio, 1 - eps, 1 + eps) * advantage)
            J /= batch_size
            L /= batch_size
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            L.backward()
            J.backward()
            optimizer_actor.step()
            optimizer_critic.step()

            soft_update_actor(actor, actor_target)
            soft_update_critic(critic, critic_target)

            with torch.no_grad():
                total_critic_loss += L.item()
                total_actor_loss += J.item()

        print_gpu_memory()

        avg_critic_loss = total_critic_loss / (e + 1)
        avg_actor_loss = total_actor_loss / (e + 1)
        epoch_pbar.set_postfix({
            'Avg Critic Loss': f'{avg_critic_loss:.4f}',
            'Avg Actor Loss': f'{avg_actor_loss:.4f}'
        })

    critic.save_model("critic_model.pt")
    actor.save_model("action_model")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:None for open-end generation.")
    train(actor_config, critic_config, 500, './MATH', 8)