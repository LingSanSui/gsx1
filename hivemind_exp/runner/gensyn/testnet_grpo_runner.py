import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

# 导入hivemind库，用于创建和管理分布式哈希表(DHT)网络
import hivemind
from datasets import Dataset
from trl import GRPOConfig, ModelConfig

# 导入链上协调器，用于管理测试网络中的节点
from hivemind_exp.chain_utils import (
    SwarmCoordinator,
)
# 导入基础GRPO运行器和参数
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
# 导入测试网络专用的GRPO训练器
from hivemind_exp.trainer.gensyn.testnet_grpo_trainer import TestnetGRPOTrainer

# 创建日志记录器
logger = logging.getLogger(__name__)


@dataclass
class TestnetGRPOArguments:
    # 以下参数互斥，只能设置其中一个
    wallet_private_key: str | None = None  # EOA钱包私钥，用于区块链身份验证
    modal_org_id: str | None = None  # Modal组织ID，用于云服务身份验证

class TestnetGRPORunner(GRPORunner):
    """
    测试网络GRPO运行器，继承自基础GRPO运行器
    用于在测试网络环境中协调和管理分布式训练节点
    """
    def __init__(self, coordinator: SwarmCoordinator) -> None:
        """
        初始化测试网络GRPO运行器
        
        参数:
            coordinator: 链上协调器实例，用于管理节点注册和引导节点发现
        """
        self.coordinator = coordinator

    def get_initial_peers(self) -> list[str]:
        """
        从链上协调器获取初始对等节点列表
        
        返回:
            引导节点地址列表，用于加入现有的DHT网络
        """
        return self.coordinator.get_bootnodes()

    def register_peer(self, peer_id):
        """
        向链上协调器注册当前节点
        
        参数:
            peer_id: 当前节点的唯一标识符
        """
        logger.info(f"正在注册节点，节点ID为: {peer_id}")
        self.coordinator.register_peer(peer_id)

    def setup_dht(self, grpo_args):
        """
        设置并启动DHT网络连接，特别适用于测试网络环境
        
        参数:
            grpo_args: GRPO参数，包含初始对等节点等网络配置
            
        返回:
            初始化并启动的DHT实例
        """
        initial_peers = grpo_args.initial_peers
        if not initial_peers:
            logger.info("无法在链上找到初始对等节点；将独立运行。")

        # 创建并启动DHT实例，使用_dht_kwargs方法构建参数
        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        logger.info(f"🐝 正在加入蜂群网络，初始对等节点 = {initial_peers}")

        # 获取节点ID并生成友好名称
        peer_id = str(dht.peer_id)
        self.name = self._get_animal_name(peer_id)
        # 向链上协调器注册当前节点
        self.register_peer(peer_id)
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
    ):
        """
        运行测试网络GRPO训练流程的主方法
        
        参数:
            model_args: 模型配置参数
            grpo_args: GRPO算法参数
            training_args: 训练配置参数
            initial_datasets_fn: 获取初始数据集的函数
        """
        # 从链上获取初始对等节点
        initial_peers = self.get_initial_peers()
        logger.info(f"从链上获取到初始对等节点: {initial_peers}")
        grpo_args.initial_peers = initial_peers
        
        # 调用父类的run方法，但使用TestnetGRPOTrainer作为训练器
        super().run(
            model_args,
            grpo_args,
            training_args,
            initial_datasets_fn,
            partial(
                TestnetGRPOTrainer,  # 使用测试网络专用的训练器
                coordinator=self.coordinator  # 传递链上协调器
            ),
        )
