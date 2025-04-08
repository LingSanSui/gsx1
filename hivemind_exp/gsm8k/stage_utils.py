import logging
import time
from collections import defaultdict
from typing import Sequence
import json
import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards
import hivemind_exp.gsm8k.stage3_rewards as stage3_rewards
from hivemind_exp.dht_utils import (
    DHT,
    HivemindNode,
    get_dht_value,
    get_outputs,
    rewards_key,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage2_samples, get_stage3_samples
from hivemind_exp.gsm8k.stage_merger import (
    Any,
    merge_stage1_question,
    merge_stage2_question,
)
from hivemind_exp.hivemind_utils import SingleStageData, StageData


# 合并前一阶段的数据集
def merged_prev_stage_datasets(
    dht: DHT,
    node: HivemindNode,
    r: int,
    s: int,
    merge_fn,
    samples_fn,
    check_interval: float = 5,
    wait_timeout: float = 10,
    log_tag=None,
):
    if not log_tag:
        log_tag = node.key

    logger = logging.getLogger(f"{__name__}:{log_tag}")

    merged_qs = []

    # 从本地和DHT检索并合并上一阶段的样本
    def get_prev_rewards():
        return get_dht_value(
            dht, key=rewards_key(r, s - 1), latest=True, beam_size=1000
        )

    prev_rewards: dict[str, Any] | None = get_prev_rewards()
    start_time = time.monotonic()
    while not prev_rewards and time.monotonic() - start_time < wait_timeout:
        logger.info(
            f"无法检索轮次 {r} 阶段 {s - 1} 的奖励；将在 {check_interval}秒后重试 "
        )
        time.sleep(check_interval)
        prev_rewards = get_prev_rewards()

    # 首先添加当前节点的本地样本
    prev_outputs: dict[str, list] = defaultdict(list)
    try:
        prev_node_outputs = get_outputs(dht, node.key, r, s - 1, node.get_stage_outputs)
        for _, outputs in prev_node_outputs.values():
            prev_outputs[node.key].append(outputs)
    except ValueError:
        # 在轮次开始后加入
        logger.info(f"无法检索轮次 {r} 阶段 {s - 1} 的本地输出")

    # 仅当奖励可用时添加其他节点的样本
    if prev_rewards:
        node_keys = prev_rewards.keys()
        for node_key in node_keys:
            if node_key == node.key:
                continue
            try:
                prev_node_outputs = get_outputs(dht, node_key, r, s - 1)
                for _, outputs in prev_node_outputs.values():
                    prev_outputs[node_key].append(outputs)
            except ValueError:
                # 跳过此节点当前轮次和阶段的答案
                logger.info(
                    f"发现节点 {node_key} 发布的奖励，但没有输出！"
                )

    # 合并所有样本
    q_to_keyed_outputs: dict[str, dict[str, Any]] = defaultdict(dict)
    for node_key, all_outputs in prev_outputs.items():
        for outputs in all_outputs:
            q_to_keyed_outputs[outputs["question"]][node_key] = outputs

    for outputs in q_to_keyed_outputs.values():
        merged = merge_fn(outputs)
        merged_qs.append(merged)
    
    # 添加日志输出merged_qs的信息
    logger.info(f"合并了 {len(merged_qs)} 个问题")
    if merged_qs:
        # 输出第一个元素作为示例
        #logger.info(f"merged_qs示例: {merged_qs[0]}")
        # 如果需要更详细的信息，可以使用以下代码
        # import json
        logger.debug(f"<-------->完整的merged_qs: {json.dumps(merged_qs, indent=2)}")

    return samples_fn(merged_qs)


# GSM8K阶段数据处理
def gsm8k_stage_data(
    dht: DHT,
    node: HivemindNode,
    initial_train_dataset,
    initial_test_dataset,
    check_interval: float = 5,
    log_tag=None,
):
    def cumulative_reward_0(**kwargs):
        return stage1_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_1(**kwargs):
        return stage2_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_2(**kwargs):
        return stage3_rewards.hivemind_cumulative_reward(node, **kwargs)

    def stage2_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage1_question,
            get_stage2_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    def stage3_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage2_question,
            get_stage3_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    # 获取轮次获胜者
    def round_winners(limit=10) -> Sequence[str]:
        final_stage_outputs, _ = merged_prev_stage_datasets(
            dht,
            node,
            node.round_num,
            3,
            lambda x: x,
            lambda v: (v, v),
            check_interval=check_interval,
            log_tag=log_tag,
        )
        rewards = defaultdict(float)
        for outputs in final_stage_outputs:
            for node_key, output in outputs.items():
                prompts = [
                    [
                        {"role": "system", "content": output["question"]},
                        {"role": "system", "content": output["stage3_prompt"]},
                    ],
                ]
                final_answer = next(iter(output["final_agent_decision"].items()))[1]
                completions = [[{"role": "assistant", "content": final_answer}]]
                cumulative_reward_2(prompts=prompts, completions=completions, **output)
                rewards[node_key] += sum(node.rewards)

        rewards = sorted(list(rewards.items()), key=lambda x: x[1], reverse=True)
        return [n for n, _ in rewards][:limit]

    return StageData(
        round_winner_fn=round_winners,
        stages=[
            SingleStageData(
                name="0",
                reward_funcs=[
                    stage1_rewards.xmlcount_reward_func,
                    stage1_rewards.soft_format_reward_func,
                    stage1_rewards.strict_format_reward_func,
                    stage1_rewards.int_reward_func,
                    stage1_rewards.correctness_reward_func,
                    cumulative_reward_0,
                ],
                datasets_fn=lambda r, s: (initial_train_dataset, initial_test_dataset),  # type: ignore
            ),
            SingleStageData(
                name="1",
                reward_funcs=[
                    stage2_rewards.proper_id_reward_func,
                    stage2_rewards.correctness_reward_func,
                    stage2_rewards.strict_format_reward_func,
                    stage2_rewards.soft_format_reward_func,
                    stage2_rewards.xmlcount_reward_func,
                    cumulative_reward_1,
                ],
                datasets_fn=stage2_datasets_fn,  # type: ignore
            ),
            SingleStageData(
                name="2",
                reward_funcs=[
                    stage3_rewards.consensus_reward_func,
                    stage3_rewards.concensus_correctness_reward_func,
                    stage3_rewards.question_recreation_reward_func,
                    stage3_rewards.final_correctness_reward_func,
                    stage3_rewards.strict_format_reward_func,
                    stage3_rewards.soft_format_reward_func,
                    stage3_rewards.xmlcount_reward_func,
                    cumulative_reward_2,
                ],
                datasets_fn=stage3_datasets_fn,  # type: ignore
            ),
        ],
    )
