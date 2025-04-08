import copy

import pytest

# 导入提示生成相关的函数和测试数据
from hivemind_exp.gsm8k.generate_prompts import *
from hivemind_exp.tests.fake_data import *


# 测试获取第二阶段样本的功能
def test_get_stage2_samples():
    # 打印第一阶段合并数据生成的第二阶段样本
    print(get_stage2_samples([STAGE_1_MERGED]))


# 测试当某些代理回答缺失时获取第二阶段样本的功能
def test_get_stage2_samples_missing_agents():
    # 复制第一阶段合并数据
    s1 = copy.deepcopy(STAGE_1_MERGED)
    s2 = copy.deepcopy(s1)
    # 删除不同代理的回答以模拟缺失情况
    del s1["agent_answers"]["0"]
    del s2["agent_answers"]["1"]
    # 测试在代理回答缺失的情况下是否仍能生成第二阶段样本
    get_stage2_samples([s1, s2])


# 测试获取第三阶段样本的功能
def test_get_stage3_samples():
    # 打印第二阶段合并数据生成的第三阶段样本
    print(get_stage3_samples([STAGE_2_MERGED]))


# 测试当某些代理意见缺失时获取第三阶段样本的功能
def test_get_stage3_samples_missing_agents():
    # 复制第二阶段合并数据
    s1 = copy.deepcopy(STAGE_2_MERGED)
    s2 = copy.deepcopy(s1)
    # 删除不同代理的意见以模拟缺失情况
    del s1["agent_opinion"][CK]
    del s2["agent_opinion"]["0"]
    # 测试在代理意见缺失的情况下是否仍能生成第三阶段样本
    get_stage3_samples([s1, s2])
