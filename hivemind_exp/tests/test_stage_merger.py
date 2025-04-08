import pytest

# 导入阶段合并相关的函数和测试数据
from hivemind_exp.gsm8k.stage_merger import *
from hivemind_exp.tests.fake_data import *


# 测试第一阶段的合并功能
def test_merge_stage1():
    # 合并第一阶段的输出数据
    merged = merge_stage1_question(STAGE_1_OUTPUTS)
    # 验证合并结果是否符合预期
    assert merged == STAGE_1_MERGED


# 测试第二阶段的合并功能
def test_merge_stage2():
    # 合并第二阶段的输出数据
    merged = merge_stage2_question(STAGE_2_OUTPUTS)
    # 验证合并结果是否符合预期
    assert merged == STAGE_2_MERGED
