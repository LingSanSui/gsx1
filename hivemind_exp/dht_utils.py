from functools import lru_cache
from typing import Any

# 导入hivemind的DHT类，用于分布式哈希表操作
from hivemind.dht import DHT
# 导入ValueWithExpiration类，用于处理带过期时间的值
from hivemind.utils import ValueWithExpiration

# 导入HivemindNode类，表示网络中的一个节点
from hivemind_exp.hivemind_utils import HivemindNode

# DHT网络中的键值定义
# 当前轮次和阶段号的键，由协调者节点发布，没有子键
# 这个键用于存储和获取当前训练的轮次和阶段信息
ROUND_STAGE_NUMBER_KEY = "rl_swarm_rs"  # 无子键。由协调者节点发布。

# 排行榜键前缀，后面会附加轮次和阶段（例如：rl_swarm_leaderboard_0_0）
# 子键为度量指标，由协调者节点发布
# 这个键用于存储和获取特定轮次和阶段的排行榜信息
LEADERBOARD_KEY_PREFIX = (
    "rl_swarm_leaderboard"  # 子键 = 度量指标。由协调者节点发布。
)

# 奖励键，子键为度量指标，所有节点都可以发布
# 这个键用于存储和获取特定轮次和阶段的奖励信息
REWARDS_KEY = "rl_swarm_rewards"  # 子键 = 度量指标。所有节点都可以发布。

# 输出键前缀，后面会附加节点键、轮次和阶段（例如：rl_swarm_outputs_abcde_0_0）
# 子键为示例哈希，所有节点都可以发布
# 这个键用于存储和获取特定节点在特定轮次和阶段的输出信息
OUTPUTS_KEY_PREFIX = "rl_swarm_outputs"  # 子键 = 示例哈希。所有节点都可以发布。


# 生成特定轮次和阶段的排行榜键
def leaderboard_key(round_num, stage) -> str:
    """
    生成特定轮次和阶段的排行榜键
    
    参数:
        round_num: 轮次号
        stage: 阶段号
        
    返回:
        格式化的排行榜键字符串，例如：rl_swarm_leaderboard_0_0
    """
    return f"{LEADERBOARD_KEY_PREFIX}_{round_num}_{stage}"


# 生成特定轮次和阶段的奖励键
def rewards_key(round_num, stage) -> str:
    """
    生成特定轮次和阶段的奖励键
    
    参数:
        round_num: 轮次号
        stage: 阶段号
        
    返回:
        格式化的奖励键字符串，例如：rl_swarm_rewards_0_0
    """
    return f"{REWARDS_KEY}_{round_num}_{stage}"


# 生成特定节点、轮次和阶段的输出键
def outputs_key(node_key: str, round_num, stage) -> str:
    """
    生成特定节点、轮次和阶段的输出键
    
    参数:
        node_key: 节点键
        round_num: 轮次号
        stage: 阶段号
        
    返回:
        格式化的输出键字符串，例如：rl_swarm_outputs_abcde_0_0
    """
    return f"{OUTPUTS_KEY_PREFIX}_{node_key}_{round_num}_{stage}"


# 根据HivemindNode对象生成输出键
def node_outputs_key(node: HivemindNode) -> str:
    """
    根据HivemindNode对象生成输出键
    
    参数:
        node: HivemindNode实例
        
    返回:
        格式化的输出键字符串
    """
    return outputs_key(node.key, node.round_num, node.stage_num)


@lru_cache
def get_outputs(
    dht: DHT, node_key: str, r, s, get_cached_fn=None
) -> dict[str, tuple[float, dict]]:  # Q: (timestamp, outputs)
    """
    获取特定节点在特定轮次和阶段的输出
    
    该函数首先尝试从本地缓存获取数据，如果缓存中没有，则从DHT网络获取。
    使用@lru_cache装饰器可以缓存函数调用结果，提高性能。
    
    参数:
        dht: DHT实例，用于与分布式哈希表交互
        node_key: 节点键
        r: 轮次号
        s: 阶段号
        get_cached_fn: 可选的缓存获取函数
        
    返回:
        字典，键为问题，值为(时间戳, 输出)元组
        
    异常:
        ValueError: 如果无法获取输出数据
    """
    # 首先尝试使用提供的缓存函数获取数据
    if get_cached_fn:
        if outputs := get_cached_fn(r, s):
            return outputs

    # 如果本地缓存中没有，则从DHT网络获取数据
    # 这是一个网络通信操作，通过DHT获取对等节点的输出
    if outputs := get_dht_value(dht, key=outputs_key(node_key, r, s), latest=False):
        return outputs

    # 如果无法获取输出，则抛出异常
    raise ValueError(
        f"could not retrieve stage outputs for {node_key} at round {r} stage {s}"
    )


def get_round_and_stage(
    dht: DHT,
) -> tuple[int, int]:
    """
    从DHT网络获取当前的轮次和阶段
    
    这个函数是网络通信的关键部分，它从分布式哈希表中获取当前训练的轮次和阶段信息。
    这些信息由协调者节点发布，所有其他节点通过此函数获取。
    
    参数:
        dht: DHT实例，用于与分布式哈希表交互
        
    返回:
        包含轮次号和阶段号的元组
        
    异常:
        ValueError: 如果无法找到当前轮次和阶段信息
    """
    # 从DHT网络获取最新的轮次和阶段值
    # 这是一个网络通信操作，通过DHT获取协调者发布的信息
    value = get_dht_value(dht, key=ROUND_STAGE_NUMBER_KEY, latest=True)
    if not value:
        raise ValueError("cannot find current round and stage")

    # 解析轮次和阶段
    round_num, stage = value
    return round_num, stage


def get_dht_value(dht: DHT, **kwargs) -> Any | None:
    """
    从DHT网络获取值的通用函数
    
    这是与DHT网络交互的核心函数，所有从分布式哈希表获取数据的操作都通过此函数进行。
    函数处理了值的包装和解包，以及子键的处理。
    
    参数:
        dht: DHT实例，用于与分布式哈希表交互
        **kwargs: 传递给dht.get()的关键字参数，通常包括key和latest
        
    返回:
        获取的值，如果没有找到则返回None
    """
    # 从DHT网络获取包装的值
    # 这是一个网络通信操作，通过DHT的get方法从分布式网络获取数据
    wrapper = dht.get(**kwargs)
    if not wrapper:
        return None

    # 确保返回的是ValueWithExpiration类型
    # DHT返回的值都是带过期时间的包装值
    assert isinstance(wrapper, ValueWithExpiration)
    value = wrapper.value
    
    # 如果值是字典类型，说明存在子键，需要解包ValueWithExpiration
    # 子键的值也是ValueWithExpiration类型，需要提取其中的value
    if isinstance(value, dict):
        # 子键存在；解包ValueWithExpiration
        return {k: v.value for k, v in value.items()}
    return value
