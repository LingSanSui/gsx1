import gc
import logging
import time
from typing import Any

import datasets
import torch
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from trl import GRPOConfig, GRPOTrainer

from hivemind_exp.dht_utils import (
    ROUND_STAGE_NUMBER_KEY,
    get_dht_value,
    get_round_and_stage,
    leaderboard_key,
    node_outputs_key,
    rewards_key,
)
from hivemind_exp.hivemind_utils import HivemindNode, StageData
from hivemind_exp.name_utils import get_name_from_peer_id


class HivemindGRPOTrainer:
    """
    基于Hivemind的GRPO训练器
    
    这是GRPOTrainer的子类，通过将中间结果发布到连接的Hivemind DHT网络来实现多阶段GRPO训练。
    它是分布式训练系统中的核心组件，负责训练模型并与其他节点共享结果。
    """

    class PublishingGRPOTrainer(GRPOTrainer):
        """
        负责将训练结果发布到DHT网络的GRPO训练器
        
        这个内部类继承自GRPOTrainer，扩展了其功能以将训练过程中的输出和奖励发布到DHT网络，
        使得其他节点可以获取这些信息。它是分布式训练系统中的网络通信核心。
        """
        def __init__(
            self,
            node: HivemindNode,
            dht: DHT,
            tokenizer,
            logger,
            **kwargs,
        ):
            """
            初始化PublishingGRPOTrainer
            
            参数:
                node: Hivemind节点实例
                dht: DHT网络实例，用于与分布式哈希表交互
                tokenizer: 分词器，用于处理文本
                logger: 日志记录器
                **kwargs: 传递给父类GRPOTrainer的其他参数
            """
            self.node = node  # Hivemind节点
            self.dht = dht    # DHT网络实例
            self.logger = logger  # 日志记录器
            self.stage_rewards = 15.0  # 当前阶段的累计奖励
            super().__init__(processing_class=tokenizer, **kwargs)

        def publish_leaderboard(self):
            """
            发布排行榜到DHT网络
            
            这个方法由协调者节点调用，用于收集所有节点的奖励信息，
            生成排序后的排行榜，并将其发布到DHT网络供所有节点查看。
            这是网络通信的关键部分，实现了节点间的竞争和协作机制。
            """
            r, s = self.node.round_num, self.node.stage_num
            # 从DHT网络获取当前轮次和阶段的所有节点奖励
            curr_rewards: dict[str, Any] | None = get_dht_value(
                self.dht, key=rewards_key(r, s), latest=True
            )
            if curr_rewards:
                # 创建按奖励值降序排序的(节点键, 奖励)对列表
                leaderboard = list(
                    sorted(
                        curr_rewards.items(), key=lambda t: (t[1], t[0]), reverse=True
                    )
                )
                # 将排行榜存储到DHT网络
                # 这是一个网络通信操作，将数据发布到分布式网络
                self.dht.store(
                    key=leaderboard_key(r, s),
                    value=leaderboard,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
            else:
                self.logger.info(f"无法获取轮次 {r} 阶段 {s - 1} 的奖励数据")

        def compute_loss(self, model, inputs, *args, **kwargs):
            """
            计算模型损失并将输出和奖励发布到DHT网络
            
            这个方法重写了父类的compute_loss方法，在计算损失的同时，
            将模型的输出和获得的奖励发布到DHT网络，使其他节点可以获取这些信息。
            这是分布式训练系统中的核心网络通信操作。
            
            参数:
                model: 模型实例
                inputs: 输入数据
                *args, **kwargs: 传递给父类方法的其他参数
                
            返回:
                计算的损失值
            """
            # 调用父类方法计算损失
            loss = super().compute_loss(model, inputs, *args, **kwargs)
            
            # 奖励函数必须保存node.outputs和node.rewards！
            # 这里的代码负责在适当的时机将数据发布到DHT网络
            
            # 获取问题和输出值
            question = self.node.outputs["question"]
            value = (time.time(), self.node.outputs)
            
            # 将输出存储到DHT网络
            # 这是一个网络通信操作，将节点的输出发布到分布式网络
            self.dht.store(
                key=node_outputs_key(self.node),  # 使用节点特定的输出键
                subkey=question,                  # 使用问题作为子键
                value=value,                      # 存储(时间戳, 输出)元组
                expiration_time=get_dht_time() + self.node.out_expiration,  # 设置过期时间
            )
            
            # 同时将输出存入本地缓存
            self.node.put_stage_outputs(
                self.node.round_num, self.node.stage_num, question, value
            )

            # 累加最新的奖励值
            self.stage_rewards += sum(self.node.rewards)
            self.logger.info(
                f"          ✅✅✅✅✅✅------✅✅✅✅✅ "
            )
            self.logger.info(
                f" key ------>> 当前key值为 {rewards_key(self.node.round_num, self.node.stage_num)}"
            )
            self.logger.info(
                f" subkey ------>> 当前subkey值为 {self.node.key}"
            )
            self.logger.info(
                f" value ------>> 当前value值为 {self.stage_rewards}"
            )
            self.logger.info(
                f" expiration_time ------>> 当前expiration_time值为 {get_dht_time() + self.node.out_expiration}"
            )
            self.logger.info(
                f"          ✅✅✅✅✅✅------✅✅✅✅✅ "
            )
            # 将累计奖励存储到DHT网络
            # 这是另一个网络通信操作，将节点的奖励发布到分布式网络
            self.dht.store(
                key=rewards_key(self.node.round_num, self.node.stage_num),  # 使用当前轮次和阶段的奖励键
                subkey=self.node.key,                                      # 使用节点键作为子键
                value=self.stage_rewards,                                  # 存储累计奖励值
                expiration_time=get_dht_time() + self.node.out_expiration,  # 设置过期时间
            )
            
            # 如果是协调者节点，则发布排行榜
            if self.node.is_coordinator:
                self.publish_leaderboard()

            return loss

    def __init__(
        self,
        node: HivemindNode,
        dht: DHT,
        stage_data: StageData,
        config: GRPOConfig,
        model,
        tokenizer,
        log_tag=None,
        **kwargs,
    ):
        # The single coordinator is responsible for incrementing round + stage numbers.
        # TODO(lou): Allow ability to choose different coordinators?
        self.node = node
        self.dht = dht

        self.stage_data = stage_data

        self.config = config
        assert self.config.output_dir
        self.config.output_dir += f"-{get_name_from_peer_id(self.node.key, True)}"  # TODO: Add animal name to save path in more appropriate spot
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not log_tag:
            log_tag = self.node.key

        self.logger = logging.getLogger(f"{__name__}:{log_tag}")

    def wait_for(self, result_fn=lambda: None, interval=10, timeout=30):
        """
        等待函数执行直到超时或返回非None结果
        
        这个辅助方法用于等待某个操作完成，比如等待DHT网络中的数据可用。
        它会定期调用result_fn函数，直到函数返回非None值或超时。
        
        参数:
            result_fn: 要执行的函数，应返回None表示继续等待，非None表示完成
            interval: 检查间隔，单位为秒
            timeout: 最大等待时间，单位为秒
            
        返回:
            result_fn的返回值，如果超时则返回None
        """
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            result = result_fn()
            if result is None:
                time.sleep(interval)
            else:
                break

        return result

    def train_stages(self, round_num, start_stage, is_coordinator):
        """
        执行特定轮次的训练阶段
        
        这个方法是训练过程的核心，负责执行从start_stage开始的所有训练阶段。
        如果是协调者节点，它还会将当前轮次和阶段信息发布到DHT网络。
        
        参数:
            round_num: 当前训练轮次
            start_stage: 开始的阶段索引
            is_coordinator: 是否为协调者节点
        """
        # TODO: Needs checkpoint loading
        self.node.round_num = round_num
        for i, stage in enumerate(self.stage_data.stages[start_stage:]):
            stage_num = start_stage + i
            self.node.stage_num = stage_num

            if is_coordinator:
                # 如果是协调者节点，将当前轮次和阶段信息发布到DHT网络
                # 这是一个网络通信操作，使其他节点能够获知当前训练进度
                self.dht.store(
                    key=ROUND_STAGE_NUMBER_KEY,
                    value=(self.node.round_num, stage_num),
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )

            self.logger.info(f"📈 训练轮次: {round_num} 阶段: {stage_num}")
            # 获取当前阶段的训练和测试数据集
            train_dataset, test_dataset = stage.datasets_fn(round_num, stage_num)
            # 准备训练器参数
            kwargs = {
                "model": self.model,
                "args": self.config,
                "reward_funcs": stage.reward_funcs,  # 当前阶段的奖励函数
                "train_dataset": train_dataset,
                "eval_dataset": test_dataset,
            }
            # 创建PublishingGRPOTrainer实例，它会将训练结果发布到DHT网络
            trainer = HivemindGRPOTrainer.PublishingGRPOTrainer(
                self.node, self.dht, self.tokenizer, self.logger, **kwargs
            )
            # 执行训练并保存模型
            self.train_and_save(trainer, train_dataset)
            self.logger.info(
                f"📉 完成训练轮次: {round_num} 阶段: {stage_num}"
            )

        # Push to HF hub if desired
        # TODO: Come back and add additional logic checking if they've provided access token+HF username
        if self.config.push_to_hub_token is not None:
            self.logger.info("正在将模型推送到 Hugging Face Hub...")
            try:
                trainer.push_to_hub(
                    tags=[
                        "rl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {get_name_from_peer_id(self.node.key)}",
                    ]
                )
                time.sleep(1)
            except Exception:
                self.logger.info(
                    "推送模型到 Hugging Face Hub 失败。当您完成训练后，请尝试按照以下说明手动推送模型：https://huggingface.co/docs/hub/en/models-uploading"
                )

        self.cleanup()

    def cleanup(self):
        """
        清理训练资源和缓存
        
        这个方法在训练阶段完成后调用，用于释放内存和清理缓存，
        包括Python垃圾回收、PyTorch缓存清理和节点阶段缓存清理。
        这对于长时间运行的分布式训练系统非常重要，可以防止内存泄漏。
        """
        # 清理各种阶段缓存
        gc.collect()  # 触发Python垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理CUDA缓存
            torch.cuda.ipc_collect()  # 清理CUDA IPC资源
        if torch.backends.mps.is_available():  # type: ignore
            torch.mps.empty_cache()  # type: ignore  # 清理MPS缓存（苹果M系列芯片）
        try:
            if torch.xpu.is_available():  # type: ignore
                torch.xpu.empty_cache()  # type: ignore  # 清理XPU缓存（Intel GPU）
        except AttributeError:
            pass

        # 清理节点的阶段缓存，释放存储的输出和奖励数据
        self.node.clear_stage_cache()

    def train_and_save(self, trainer, train_dataset):
        """
        执行训练并保存模型和指标
        
        这个方法使用提供的训练器执行训练，然后记录和保存训练指标，
        最后保存模型和分词器。在分布式环境中，它还会等待所有进程完成加载。
        
        参数:
            trainer: PublishingGRPOTrainer实例，用于执行训练
            train_dataset: 训练数据集，用于计算样本数量
        """
        # 执行训练并获取结果
        train_result = trainer.train()

        # 记录和保存指标
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)  # 添加训练样本数量到指标
        trainer.log_metrics("train", metrics)  # 记录训练指标
        trainer.save_metrics("train", metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练器状态

        # 保存模型
        self.logger.info("正在保存模型")
        trainer.model.config.use_cache = True  # 启用模型缓存以提高推理性能
        trainer.save_model(self.config.output_dir)  # 保存模型到输出目录
        self.logger.info(f"模型已保存到 {self.config.output_dir}")
        
        # 在分布式环境中等待所有进程完成
        assert self.config.distributed_state
        self.config.distributed_state.wait_for_everyone()  # 等待所有进程加载完成

        # 保存分词器
        self.tokenizer.save_pretrained(self.config.output_dir)
        self.logger.info(f"分词器已保存到 {self.config.output_dir}")

    def get_round_and_stage(self):
        """
        获取当前的轮次和阶段
        
        这个方法是对dht_utils.get_round_and_stage的包装，用于从DHT网络获取
        当前的训练轮次和阶段信息。这些信息由协调者节点发布，所有跟随者节点
        通过此方法获取以同步训练进度。
        
        返回:
            包含轮次号和阶段号的元组
            
        异常:
            ValueError: 如果无法从DHT网络获取轮次和阶段信息
        """
        return get_round_and_stage(self.dht)

    def coordinator_train(self):
        """
        协调者节点的训练入口
        
        这个方法由协调者节点调用，负责管理整个训练过程。
        协调者节点从轮次0开始，依次执行每个轮次的所有阶段，并将轮次和阶段信息
        发布到DHT网络，使跟随者节点能够同步训练进度。
        
        训练会在达到最大轮次数或超时时结束。
        """
        round_num = 0  # 从轮次0开始
        start_time = time.monotonic()  # 记录开始时间
        # 在未达到最大轮次且未超时的情况下继续训练
        while (
            round_num < self.stage_data.max_rounds
            and time.monotonic() - start_time < self.stage_data.train_timeout
        ):
            self.logger.info(f"🤖 开始新的训练轮次: {round_num}")

            # 获取可见的多地址，确保DHT网络连接正常
            _ = self.dht.get_visible_maddrs(latest=True)
            # 执行当前轮次的所有阶段，从阶段0开始
            self.train_stages(round_num, 0, is_coordinator=True)

            # 轮次完成，准备下一轮
            round_num += 1
            if round_num == self.stage_data.max_rounds:
                return  # 达到最大轮次，训练结束

        # 如果是因为超时而退出循环，记录日志
        self.logger.info("训练已超时！")

    def follower_train(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=30.0
    ):
        """
        跟随者节点的训练入口
        
        这个方法由跟随者节点调用，负责跟随协调者节点的训练进度。
        跟随者节点定期从DHT网络获取当前的轮次和阶段信息，然后执行相应的训练阶段。
        为了避免重复训练，它会记录已完成的轮次，并使用指数退避策略减少对已完成轮次的检查频率。
        
        参数:
            check_interval: 检查DHT网络的初始间隔，单位为秒
            log_timeout: 日志记录超时，避免频繁记录相同错误
            max_check_interval: 最大检查间隔，单位为秒，用于指数退避策略
        """
        done_rounds = set()  # 记录已完成的轮次
        start_time = time.monotonic()  # 记录开始时间
        fetch_log_time = start_time  # 上次获取日志的时间
        check_backoff = check_interval  # 检查间隔，用于已完成轮次的指数退避
        
        # 在未超时的情况下继续训练
        while time.monotonic() - start_time < self.stage_data.train_timeout:
            curr_time = time.monotonic()
            # 获取可见的多地址，确保DHT网络连接正常
            _ = self.dht.get_visible_maddrs(latest=True)

            # 从DHT网络获取当前轮次和阶段
            try:
                round_num, stage = self.get_round_and_stage()
            except Exception as e:
                # 如果无法获取轮次和阶段信息，记录日志并继续尝试
                if curr_time - fetch_log_time > log_timeout:
                    self.logger.debug(
                        f"无法获取轮次和阶段信息: {e}。将在 {check_interval}秒 后重新检查。"
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            # 如果是新的轮次（未完成），则开始训练
            if round_num not in done_rounds:
                self.logger.info(
                    f"🐝 加入训练轮次: {round_num} 从阶段: {stage} 开始"
                )
                try:
                    # 从当前阶段开始训练
                    self.train_stages(round_num, stage, is_coordinator=False)
                except datasets.exceptions.DatasetGenerationError:
                    # 如果数据集生成错误且不是第一阶段，尝试从阶段0重新开始
                    if stage > 0:
                        self.logger.info("正在从阶段0重新开始训练！")

                        # 从阶段0重新开始
                        self.train_stages(round_num, 0, is_coordinator=False)
                    else:
                        raise  # 如果是第一阶段出错，则抛出异常

                # 标记轮次为已完成
                done_rounds.add(round_num)
                check_backoff = check_interval  # 重置退避间隔
            else:
                # 如果轮次已完成，使用指数退避策略减少检查频率
                if check_backoff != 30:
                    self.logger.info(
                        f":{self.node.key}:已完成训练轮次: {round_num}。将在 {check_backoff}秒 后重新检查是否有新任务，日志暂停刷新，不是卡住，耐心等待。"
                    )
                time.sleep(check_backoff)
                # 指数退避：将检查间隔翻倍，但不超过最大间隔
                check_backoff = min(check_backoff * 2, max_check_interval)

            # 如果达到最后一轮，训练结束
            if round_num == self.stage_data.max_rounds - 1:
                return

        # 如果是因为超时而退出循环，记录日志
        self.logger.info("训练已超时！")

    def train(self):
        """
        训练入口方法
        
        这是HivemindGRPOTrainer类的主要入口方法，根据节点类型（协调者或跟随者）
        调用不同的训练流程。协调者节点负责管理整个训练过程，而跟随者节点则跟随
        协调者的进度进行训练。
        
        该方法捕获并打印所有异常，确保训练过程中的错误不会导致整个程序崩溃。
        """
        try:
            # 根据节点类型选择不同的训练流程
            if self.node.is_coordinator:
                # 协调者节点：管理训练进度，发布轮次和阶段信息
                self.coordinator_train()
            else:
                # 跟随者节点：跟随协调者的进度进行训练
                self.follower_train()

        except Exception:
            # 捕获并打印所有异常，确保训练过程中的错误不会导致程序崩溃
            import traceback

            traceback.print_exc()
