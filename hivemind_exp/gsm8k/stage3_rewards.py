import os
import random
import re
from difflib import SequenceMatcher

import numpy as np

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode


# 从文本中提取XML多数意见标识
def extract_xml_identity(text: str) -> str:
    id = text.split("<majority>")[-1]
    id = id.split("</majority>")[0]
    return id.strip()


# 从文本中提取XML最终答案
def extract_xml_final_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# 从文本中提取XML问题
def extract_xml_question(text: str) -> str:
    question = text.split("<question>")[-1]
    question = question.split("</question>")[0]
    return question.strip()


# 从文本中提取XML ID
def extract_xml_ids(text: str) -> str:
    ids = []
    ids_raw = text.split("<student>")[1:]
    for id in ids_raw:
        ids += [id.split("</student>")[0].strip()]
    return ids


# 从文本中提取XML选择标识
def extract_xml_choices(text: str) -> str:
    ids = []
    ids_raw = text.split("<identify>")[1:]
    for id in ids_raw:
        ids += [id.split("</identify>")[0].strip()]
    return ids


# 从文本中提取原始问题
def extract_original_question(text: str) -> str:
    q = text.split("  \n\nThe following answers to this question were suggested:")[0]
    q = q.split("The question we were given is: ")[-1]
    return q.strip()


# 从文本中提取答案
def extract_answers(text: str) -> str:
    answers = {}
    raw = text.split(
        "  \nAfter comparing these answers, the following feedback was given about which answer is best: \n"
    )[0].split("<student>")[1:]
    for a in raw:
        id = a.split("</student>")[0].strip()
        ans = a.split("</student> 说 \n")[-1].strip()
        answers[id] = ans
    return answers


# 计算XML标签的使用情况并给予奖励分数
def count_xml(text) -> float:
    count = 0.0
    if text.count("<summarize_feedback>\n") == 1:
        count += 0.125
    if text.count("\n</summarize_feedback>\n") == 1:
        count += 0.125
    if text.count("<majority>\n") == 1:
        count += 0.125
    if text.count("\n</majority>\n") == 1:
        count += 0.125
    if text.count("<question>\n") == 1:
        count += 0.125
    if text.count("\n</question>\n") == 1:
        count += 0.125
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


# 计算群体多数意见
def swarm_majority(choices):
    votes = {}
    max_votes = 0
    for c in choices:
        if c in votes:
            votes[c] += 1
        else:
            votes[c] = 1
        if votes[c] > max_votes:
            max_votes = votes[c]
    majority = []
    for c in votes:
        if votes[c] >= max_votes:
            majority += [c]
    return majority


# 奖励函数
# 共识奖励函数 - 检查选择是否与多数意见一致
def consensus_reward_func(
    prompts, completions, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    critic_choices = extract_xml_choices(p)
    majority_choices = swarm_majority(critic_choices)
    extracted_responses = [extract_xml_identity(r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "consensus_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nCritic Choice Distribution:\n{critic_choices}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in majority_choices}"
            f.write(out_line)
    return [
        1.0 * weighting if r in majority_choices else 0.0 for r in extracted_responses
    ]


# 问题重建奖励函数 - 评估问题重建的质量
def question_recreation_reward_func(
    prompts, completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    q = extract_original_question(p)
    recreated_qs = [extract_xml_question(r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "question_recreation_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nOriginal Question:\n{q}\n\nExtracted recreation:\n{recreated_qs[0]}\n\nGot reward? {SequenceMatcher(None, recreated_qs[0], q).ratio()}"
            f.write(out_line)
    return [SequenceMatcher(None, r, q).ratio() * weighting for r in recreated_qs]


# 共识正确性奖励函数 - 评估选择的答案的正确性
def concensus_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    agent_answers = extract_answers(p)
    extracted_responses = [extract_xml_identity(r) for r in responses]
    chosen_rewards = []
    for r in extracted_responses:
        cur_reward = 0
        if r in agent_answers:
            if stage1_rewards.extract_xml_answer(agent_answers[r]) == answer[0]:
                cur_reward += 1.0
            if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                cur_reward += 0.5
            pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            cur_reward += stage1_rewards.count_xml(agent_answers[r])
        elif r in [
            "None",
            "No one",
            "All answers are wrong",
            "All answers were wrong",
            "All are wrong",
            "All were wrong",
            "None are correct",
            "None were correct",
            "No one is correct",
        ]:
            agent_as = [
                stage1_rewards.extract_xml_answer(agent_answers[id])
                for id in agent_answers
            ]
            check_submissions = [
                True if r == a else False for r, a in zip(agent_as, answer)
            ]
            if all(check_submissions):
                cur_reward += 10
        chosen_rewards += [cur_reward]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        if extracted_responses[0] in agent_answers:
            os.makedirs(
                f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                exist_ok=True,
            )
            log_file = os.path.join(
                "model_output_samples",
                f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                "correctness_samps.txt",
            )
            with open(log_file, "a") as f:
                f.write("-" * 20)
                out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nChosen answer ID:\n{extracted_responses[0]}\n\nExtracted:\n{agent_answers[extracted_responses[0]]}\n\nReward for choice: {chosen_rewards[0]}"
                f.write(out_line)
    return [r * weighting for r in chosen_rewards]


# 最终正确性奖励函数 - 评估最终答案的正确性
def final_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_final_answer(r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "final_answer_correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Prompt:\n{p}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    return [
        1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


# 严格格式奖励函数 - 检查完成是否具有特定格式
def strict_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    """检查完成是否具有特定格式的奖励函数。"""
    pattern = r"^<summarize_feedback>\n.*?\n</summarize_feedback>\n<majority>\n.*?\n</majority>\n<question>\n.*?\n</question>\n<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s3_strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [1.0 * weighting if match else 0.0 for match in matches]


# 宽松格式奖励函数 - 检查完成是否具有宽松的特定格式
def soft_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    """检查完成是否具有特定格式的奖励函数。"""
    pattern = r"<summarize_feedback>.*?</summarize_feedback>\s*<majority>.*?</majority>\s*<question>.*?</question>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s3_soft_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [1.0 * weighting if match else 0.0 for match in matches]


# XML计数奖励函数 - 根据XML标签的使用情况给予奖励
def xmlcount_reward_func(
    completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    if (random.random() < 0.01) and logging:  # 1%的概率将样本写入文件
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "count_xml_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = (
                f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
            )
            f.write(out_line)
    return [count_xml(c) * weighting for c in contents]


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
    consensus_reward = consensus_reward_func(prompts, completions, logging=logging)
    concensus_correctness = concensus_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    question_recreation_reward = question_recreation_reward_func(
        prompts, completions, logging=logging
    )
    final_correctness = final_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    total_reward = [
        sum(tup)
        for tup in zip(
            consensus_reward,
            concensus_correctness,
            question_recreation_reward,
            final_correctness,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    prompt = prompts[0][-1]["content"]
    question = extract_original_question(prompt)
    if output_signal_selector == "max":
        # 生成输出行
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": question,
            "answer": answer[0],
            "stage3_prompt": prompt,
            "final_agent_decision": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
