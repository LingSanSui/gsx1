# 导入从对等节点ID获取名称和搜索对等节点ID的函数
from hivemind_exp.name_utils import get_name_from_peer_id, search_peer_ids_for_name

# 测试用的对等节点ID列表
TEST_PEER_IDS = [
    "QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N",
    "Qma9T5YraSnpRDZqRR4krcSJabThc8nwZuJV3LercPHufi",
    "Qmb8wVVVMTRmG4U1tCdaCCqietuWwpGRSbL53PA5azBViP",
]


# 测试从对等节点ID获取名称的功能
def test_get_name_from_peer_id():
    # 从测试ID列表中获取对应的名称
    names = [get_name_from_peer_id(peer_id) for peer_id in TEST_PEER_IDS]
    # 验证生成的名称是否符合预期
    assert names == [
        "thorny fishy meerkat",
        "singing keen cow",
        "toothy carnivorous bison",
    ]
    # 测试带下划线参数的名称生成
    assert get_name_from_peer_id(TEST_PEER_IDS[-1], True) == "toothy_carnivorous_bison"


# 测试根据名称搜索对等节点ID的功能
def test_search_peer_ids_for_name():
    # 测试不同名称的搜索结果
    names = ["none", "not an animal", "toothy carnivorous bison"]
    results = [search_peer_ids_for_name(TEST_PEER_IDS, name) for name in names]
    # 验证搜索结果是否符合预期
    assert results == [None, None, "Qmb8wVVVMTRmG4U1tCdaCCqietuWwpGRSbL53PA5azBViP"]
