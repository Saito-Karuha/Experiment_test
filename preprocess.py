import os
import json
import random
import re
from typing import Dict, List, Optional, Union


def extract_ground_truth(solution: str) -> Optional[str]:
    # 使用平衡括号的方法提取内容
    def find_matching_brace(s: str, start: int) -> int:
        """找到匹配的右括号位置"""
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

        # 找到 \boxed{ 的位置
    match = re.search(r'\\boxed\{', solution)
    if not match:
        return None

    start_pos = match.end()  # '{'后的位置
    end_pos = find_matching_brace(solution, start_pos)

    if end_pos == -1:
        return None

    return solution[start_pos:end_pos]


def load_math_dataset(dataset_dir: str) -> Dict[str, List[dict]]:
    """
    加载整个MATH数据集到内存中

    Args:
        dataset_dir: 数据集根目录路径

    Returns:
        包含'test'和'train'两个键的字典，每个键对应一个问题列表
    """
    dataset = {
        "test": [],
        "train": []
    }

    for split in ["test", "train"]:
        split_path = os.path.join(dataset_dir, split)

        # 确保目录存在
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            continue

            # 遍历所有主题文件夹
        for subtopic in os.listdir(split_path):
            subtopic_path = os.path.join(split_path, subtopic)

            # 确保是目录
            if not os.path.isdir(subtopic_path):
                continue

                # 获取所有json文件
            json_files = [f for f in os.listdir(subtopic_path) if f.endswith('.json')]

            for json_file in json_files:
                file_path = os.path.join(subtopic_path, json_file)
                try:
                    with open(file_path, encoding='utf-8') as f:
                        problem_data = json.load(f)

                        # 确保数据包含所需的字段
                        if all(key in problem_data for key in ["problem", "level", "type", "solution"]):
                            problem = {
                                "problem": problem_data["problem"],
                                "level": problem_data["level"],
                                "type": problem_data["type"],
                                "solution": problem_data["solution"],
                                "ground_truth": extract_ground_truth(problem_data["solution"]),
                                "source": f"{split}/{subtopic}/{json_file}"
                            }
                            dataset[split].append(problem)
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    continue

                    # 打印数据集统计信息
    print(f"Dataset loaded successfully:")
    print(f"Train set size: {len(dataset['train'])} problems")
    print(f"Test set size: {len(dataset['test'])} problems")

    return dataset


def select_random_problem(
        dataset: Dict[str, List[dict]],
        split: str = "train",
        num_problems: int = 1
) -> List[dict]:
    """
    从指定数据集分割中随机选择问题

    Args:
        dataset: 包含'test'和'train'的数据集字典
        split: 'test'或'train'
        num_problems: 要选择的问题数量

    Returns:
        选中问题的列表
    """
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

    if not dataset[split]:
        raise ValueError(f"No problems found in {split} split")

        # 确保请求的问题数量不超过可用问题数量
    num_problems = min(num_problems, len(dataset[split]))

    # 随机选择问题
    selected_problems = random.sample(dataset[split], num_problems)

    return selected_problems


# 使用示例
if __name__ == "__main__":
    # 初始化数据集（只需要执行一次）
    dataset_directory = "E:\\数据集\\MATH\\MATH"
    math_dataset = load_math_dataset(dataset_directory)

    # 示例1：从训练集中随机选择3个问题
    train_problems = select_random_problem(math_dataset, split="train", num_problems=3)
    print("\nSelected training problems:")
    for i, problem in enumerate(train_problems, 1):
        print(f"\nProblem {i}:")
        print(f"Source: {problem['source']}")
        print(f"Problem: {problem['problem']}")
        print(f"Type: {problem['type']}")
        print(f"Level: {problem['level']}")
        print(f"Ground Truth: {problem['ground_truth']}")
        print("-" * 80)

        # 示例2：从测试集中随机选择2个问题
    test_problems = select_random_problem(math_dataset, split="test", num_problems=2)
    print("\nSelected test problems:")
    for i, problem in enumerate(test_problems, 1):
        print(f"\nProblem {i}:")
        print(f"Source: {problem['source']}")
        print(f"Problem: {problem['problem']}")
        print(f"Type: {problem['type']}")
        print(f"Level: {problem['level']}")
        print(f"Ground Truth: {problem['ground_truth']}")
        print("-" * 80)