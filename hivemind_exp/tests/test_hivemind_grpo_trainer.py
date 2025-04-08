# 导入必要的库和模块
import itertools
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hivemind
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from hivemind_exp.dht_utils import (
    HivemindNode,
    leaderboard_key,
    outputs_key,
    rewards_key,
)
from hivemind_exp.hivemind_utils import SingleStageData, StageData
from hivemind_exp.tests.fake_data import CK, QUESTION, RSK, SAMPLES
from hivemind_exp.trainer.hivemind_grpo_trainer import (
    HivemindGRPOTrainer,
    get_dht_value,
)


# 模拟奖励函数，为不同节点分配不同的奖励值
def dummy_reward_func(node: HivemindNode, prompts, completions, **kwargs) -> list[int]:
    node.outputs = {"question": prompts[0][-1]["content"]}
    if node.is_coordinator:
        rewards = [2]  # 协调器节点获得2分
    else:
        rewards = [1]  # 普通节点获得1分

    node.rewards = rewards
    return rewards


# 测试模型名称
TEST_MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


# 获取模型和配置
def get_model_config(tmp_path, max_steps):
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    config = GRPOConfig(
        output_dir=tmp_path,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
    )
    return model, config


# 创建DHT和训练器
def create_dht_and_trainer(tmp_path, node, stage_data, max_steps=1, initial_peers=[]):
    dht = hivemind.DHT(start=True, initial_peers=initial_peers, cache_nearest=2)
    model, config = get_model_config(tmp_path, max_steps=max_steps)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    trainer = HivemindGRPOTrainer(
        dht=dht,
        node=node,
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage_data=stage_data,
    )
    return dht, trainer


###############
# 单节点测试 #
###############


# 测试单节点单阶段训练
def test_single_node_single_stage(tmp_path):
    node = HivemindNode.coordinator("test", CK)

    def reward_func(**kwargs):
        return dummy_reward_func(node, **kwargs)

    dht, trainer = create_dht_and_trainer(
        tmp_path,
        node,
        StageData(
            max_rounds=1,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
            ],
        ),
    )
    trainer.train()


# 测试单节点多阶段训练
def test_single_node_multi_stage(tmp_path):
    """冒烟测试：不进行实际合并，只标记完成情况。"""
    completions = {}

    def datasets_one(r, s):
        completions["merged_0"] = True
        return SAMPLES, SAMPLES

    node = HivemindNode.coordinator("test", CK)

    def reward_func(**kwargs):
        return dummy_reward_func(node, **kwargs)

    dht, trainer = create_dht_and_trainer(
        tmp_path,
        node,
        StageData(
            max_rounds=1,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
                SingleStageData(
                    name="1",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_one,  # type: ignore
                ),
            ],
        ),
    )
    trainer.train()

    # 验证第一阶段已完成合并
    assert completions == {"merged_0": True}


##############
# 多节点测试 #
##############

# 这些测试将实际检查DHT输出、奖励和排行榜。

# TODO: 修复以下测试的不稳定性问题

# 测试多节点单阶段训练
def test_multi_node_single_stage(tmp_path):
    max_rounds = 1
    max_steps = 2

    def create_stage_data(node):
        def reward_func(**kwargs):
            return dummy_reward_func(node, **kwargs)

        return StageData(
            max_rounds=max_rounds,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
            ],
        )

    # 创建协调器节点和普通节点
    node0 = HivemindNode.coordinator("test", CK)
    node1 = HivemindNode("test", "0")

    # 创建两个DHT和训练器
    dht0, trainer0 = create_dht_and_trainer(
        Path(tmp_path) / "0", node0, create_stage_data(node0), max_steps
    )
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        create_stage_data(node1),
        max_steps,
        dht0.get_visible_maddrs(),
    )
    # 并行训练两个节点
    with ThreadPoolExecutor() as executor:
        for trainer in (trainer0, trainer1):
            executor.submit(trainer.train)

    # 验证轮次和阶段信息
    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (max_rounds - 1, 0)

    # 验证输出、奖励和排行榜
    for r, s in itertools.product([0], [0]):
        # 检查输出
        outputs = get_dht_value(dht0, key=outputs_key(node0.key, r, s), latest=True)
        assert outputs
        assert outputs[QUESTION][1] == {"question": QUESTION}

        # 检查奖励
        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=True)
        assert rewards
        assert len(rewards) == 2
        assert math.isclose(rewards[CK], 2.0 * max_steps)
        assert math.isclose(rewards[node1.key], max_steps)

        # 检查排行榜
        leaderboard = get_dht_value(dht0, key=leaderboard_key(r, s), latest=True)
        assert leaderboard
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == CK
        assert math.isclose(leaderboard[0][1], 2.0 * max_steps)


# 测试多节点多阶段训练
def test_multi_node_multi_stage(tmp_path):
    """冒烟测试：不进行实际合并，只标记完成情况。"""
    completions = defaultdict(int)
    max_rounds = 2
    max_steps = 2

    def datasets_one(r, s):
        completions["merged_0"] += 1
        return SAMPLES, SAMPLES

    def datasets_two(r, s):
        completions["merged_1"] += 1
        return SAMPLES, SAMPLES

    def create_stage_data(node):
        def reward_func(**kwargs):
            return dummy_reward_func(node, **kwargs)

        return StageData(
            max_rounds=max_rounds,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
                SingleStageData(
                    name="1",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_one,  # type: ignore
                ),
                SingleStageData(
                    name="2",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_two,  # type: ignore
                ),
            ],
        )

    # 创建协调器节点和普通节点
    node0 = HivemindNode.coordinator("test", CK)
    node1 = HivemindNode("test", "0")

    # 创建两个DHT和训练器
    dht0, trainer0 = create_dht_and_trainer(
        Path(tmp_path) / "0", node0, create_stage_data(node0), max_steps
    )
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        create_stage_data(node1),
        max_steps,
        dht0.get_visible_maddrs(),
    )
    # 并行训练两个节点
    with ThreadPoolExecutor() as executor:
        for trainer in (trainer0, trainer1):
            executor.submit(trainer.train)

    # 验证轮次和阶段信息
    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (max_rounds - 1, 2)

    # 验证合并完成情况
    assert completions == {
        "merged_0": max_rounds * 2,
        "merged_1": max_rounds * 2,
    }

    # 验证输出、奖励和排行榜
    for r, s in itertools.product(range(1), range(3)):
        # 检查输出
        outputs = get_dht_value(dht0, key=outputs_key(node0.key, r, s), latest=False)
        assert outputs
        assert outputs[QUESTION][1] == {"question": QUESTION}

        # 检查奖励
        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=False)
        assert rewards
        assert len(rewards) == 2
        assert math.isclose(rewards[CK], 2.0 * max_steps)
        assert math.isclose(rewards[node1.key], max_steps)

        # 检查排行榜
        leaderboard = get_dht_value(dht0, key=leaderboard_key(r, s), latest=False)
        assert leaderboard
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == CK
        assert math.isclose(leaderboard[0][1], 2.0 * max_steps)
