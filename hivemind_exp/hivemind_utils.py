from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch


@dataclass
class HivemindNode:
    """
    表示Hivemind网络中的一个节点
    
    这个类封装了节点的元数据、状态和缓存机制，是分布式训练系统中的基本单元。
    节点可以是普通节点或协调者节点，协调者节点负责管理训练轮次和阶段。
    """
    # 节点元数据
    model_name: str  # 模型名称，表示节点使用的模型
    key: str  # 节点唯一标识符，设置为DHT的PeerID

    is_coordinator: bool = False  # 是否为协调者节点，协调者负责管理训练进程

    # 上一次训练步骤的问答输出，存储模型生成的输出
    outputs: dict[Any, Any] = field(default_factory=dict)
    
    # 轮次缓存，用于存储不同轮次和阶段的输出
    # 缓存格式：(轮次, 阶段): 问题: (时间戳, 输出)
    # 这个缓存减少了对DHT网络的请求，提高了性能
    round_cache: dict[tuple[int, int], dict[str, tuple[float, dict]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    # 上一次训练的奖励输出，存储模型获得的奖励值
    rewards: Sequence[float | int] = field(default_factory=list)

    # 由协调者递增的值，表示当前训练的进度
    round_num: int = 0  # 当前轮次，由协调者更新
    stage_num: int = 0  # 当前阶段，由协调者更新

    # 输出在DHT网络中的过期时间，单位为秒
    # 默认为4小时，防止DHT网络中存储过多过期数据
    out_expiration: int = 60 * 60 * 4  # 输出过期时间（4小时）

    @staticmethod
    def coordinator(*args, **kwargs):
        """
        创建一个协调者节点实例
        
        协调者节点负责管理训练进程，包括更新轮次和阶段，发布排行榜等。
        
        参数:
            *args: 传递给HivemindNode构造函数的位置参数
            **kwargs: 传递给HivemindNode构造函数的关键字参数
            
        返回:
            配置为协调者的HivemindNode实例
        """
        return HivemindNode(*args, **kwargs, is_coordinator=True)

    def get_stage_outputs(self, r, s) -> dict[str, tuple[float, dict]] | None:
        """
        获取特定轮次和阶段的输出
        
        从本地缓存中获取特定轮次和阶段的输出，避免从DHT网络重复获取数据。
        
        参数:
            r: 轮次号
            s: 阶段号
            
        返回:
            如果缓存中存在该轮次和阶段的输出，则返回输出字典；否则返回None
            输出字典的格式为：{问题: (时间戳, 输出)}
        """
        key = (r, s)
        if key in self.round_cache:
            return self.round_cache[key]

    def put_stage_outputs(self, r, s, question, value: tuple[float, dict]):
        """
        将输出存入特定轮次和阶段的缓存
        
        将模型生成的输出存入本地缓存，以便后续使用，减少对DHT网络的请求。
        
        参数:
            r: 轮次号
            s: 阶段号
            question: 问题标识符
            value: (时间戳, 输出)元组，包含输出的时间戳和实际输出内容
        """
        self.round_cache[(r, s)][question] = value

    def clear_stage_cache(self):
        """
        清除所有阶段缓存
        
        在训练完成或需要释放内存时调用此方法清除缓存。
        """
        self.round_cache.clear()


# 类型定义，用于提高代码的类型安全性和可读性

# 数据集函数类型：接收轮次和阶段，返回训练和测试数据集的元组
# 这种函数用于为特定轮次和阶段生成或获取数据集
DatasetsFn = Callable[
    [int, int], tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
]

# 合并函数类型：将列表合并为字典
# 这种函数用于合并来自不同节点的数据
MergeFn = Callable[[list], dict[str, dict]]

# 损失函数类型：计算列表的损失值
# 这种函数用于计算模型输出的损失或奖励
LossFn = Callable[[list], dict[str, float]]


@dataclass
class SingleStageData:
    """
    单个训练阶段的数据
    
    这个类定义了训练过程中单个阶段的数据和行为，包括阶段名称、奖励函数和数据集获取函数。
    在多阶段训练中，每个阶段可能有不同的奖励函数和数据集。
    """
    name: str  # 阶段名称，用于标识和日志记录
    reward_funcs: list[Callable]  # 奖励函数列表，用于计算模型输出的奖励
    datasets_fn: DatasetsFn  # 用于获取训练/测试数据集的函数，接收轮次和阶段参数


@dataclass
class StageData:
    """
    多阶段训练数据
    
    这个类定义了整个训练过程的多个阶段，包括各阶段的数据、获胜者确定函数和超时设置。
    它是分布式训练系统中组织训练流程的核心数据结构。
    """
    stages: Sequence[SingleStageData]  # 各个阶段的数据序列，按顺序执行
    round_winner_fn: Callable  # 确定轮次获胜者的函数，用于选择表现最好的节点

    max_rounds: int = 100  # 最大训练轮次数，防止无限训练
    train_timeout: int = 60 * 60 * 24 * 4  # 训练总超时时间（4天），单位为秒
    round_timeout: int = 60 * 60 * 4  # 单个轮次的超时时间（4小时），单位为秒

    def __len__(self):
        """
        返回阶段数量
        
        返回:
            训练过程中的阶段总数
        """
        return len(self.stages)
