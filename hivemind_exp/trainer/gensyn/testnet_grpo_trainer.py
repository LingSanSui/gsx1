from typing import Sequence

from hivemind_exp.chain_utils import SwarmCoordinator
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer


class TestnetGRPOTrainer(HivemindGRPOTrainer):
    """
    测试网络版本的GRPO训练器
    
    这个类继承自HivemindGRPOTrainer，添加了与区块链交互的功能，
    包括提交获胜者信息和从区块链获取轮次与阶段信息。
    它是连接Hivemind分布式训练系统与区块链网络的桥梁。
    """
    def __init__(self, coordinator: SwarmCoordinator, **kwargs) -> None:
        """
        初始化测试网络GRPO训练器
        
        参数:
            coordinator: 区块链协调器实例，用于与区块链网络交互
            **kwargs: 传递给父类HivemindGRPOTrainer的其他参数
        """
        self.coordinator = coordinator
        super().__init__(**kwargs)

    def submit_winners(self, round_num: int, winners: Sequence[str]):
        """
        向区块链提交轮次获胜者信息
        
        参数:
            round_num: 当前训练轮次
            winners: 获胜者ID序列，通常是表现最好的节点
        """
        self.logger.info(f"🏆 正在为第{round_num}轮提交获胜者: {winners}")
        self.coordinator.submit_winners(round_num, winners[:1])

    def get_round_and_stage(self):
        """
        从区块链获取当前轮次和阶段信息
        
        返回:
            当前轮次和阶段的元组(round_num, stage_num)
        """
        return self.coordinator.get_round_and_stage()

    def train_stages(self, round_num, start_stage, is_coordinator):
        """
        执行训练阶段并提交获胜者
        
        这个方法重写了父类的train_stages方法，在完成所有训练阶段后，
        会自动提交当前轮次的获胜者到区块链网络。
        
        参数:
            round_num: 当前训练轮次
            start_stage: 开始的阶段索引
            is_coordinator: 是否为协调者节点
        """
        super().train_stages(round_num, start_stage, is_coordinator)
        self.submit_winners(round_num, self.stage_data.round_winner_fn())

    def train(self):
        """
        执行训练过程
        
        这个方法作为训练的入口点，调用follower_train方法开始训练，
        并捕获可能发生的异常，确保训练过程的稳定性。
        """
        try:
            self.follower_train()

        except Exception:
            import traceback

            traceback.print_exc()
