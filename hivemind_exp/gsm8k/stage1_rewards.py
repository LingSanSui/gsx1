import os
import random
import re

import numpy as np

from hivemind_exp.hivemind_utils import HivemindNode


# 从文本中提取XML格式的答案
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# 计算XML标签的使用情况并给予奖励分数
def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# 奖励函数
# 正确性奖励函数 - 检查答案是否正确
def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Question:\n{q}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    return [
        1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


# 整数奖励函数 - 检查答案是否为整数
def int_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 * weighting if r.isdigit() else 0.0 for r in extracted_responses]


# 严格格式奖励函数 - 检查完成是否具有特定格式
def strict_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """检查完成是否具有特定格式的奖励函数。"""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 * weighting if match else 0.0 for match in matches]


# 宽松格式奖励函数 - 检查完成是否具有宽松的特定格式
def soft_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """检查完成是否具有特定格式的奖励函数。"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 * weighting if match else 0.0 for match in matches]


# XML计数奖励函数 - 根据XML标签的使用情况给予奖励
def xmlcount_reward_func(completions, weighting=1.0, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) * weighting for c in contents]

# Top-K累积奖励函数 - 用于提示生成的top-k选择器
def top_k_cumulative_reward(
    prompts,
    completions,
    answer,
    logging=False,
    **kwargs,
) -> list[float]:
    """
    累积所有奖励为一个总分，用于提示生成的top-k选择器
    """
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    int_reward = int_reward_func(completions)
    strict_format_reward = strict_format_reward_func(completions)
    soft_format_reward = soft_format_reward_func(completions)
    xmlcount_reward = xmlcount_reward_func(completions)
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    return total_reward


# Hivemind累积奖励函数 - 累积所有奖励并保存JSON到node.outputs
def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    """
    累积所有奖励为一个总分，并将JSON保存到node.outputs
    """
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    int_reward = int_reward_func(completions)
    strict_format_reward = strict_format_reward_func(completions)
    soft_format_reward = soft_format_reward_func(completions)
    xmlcount_reward = xmlcount_reward_func(completions)
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    if output_signal_selector == "max":
        # 生成输出行
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": prompts[0][-1]["content"],
            "answer": answer[0],
            "agent_answers": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
