import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Tuple

# 导入hivemind库，用于创建和管理分布式哈希表(DHT)网络
import hivemind
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, ModelConfig

from hivemind_exp.gsm8k.stage_utils import gsm8k_stage_data
from hivemind_exp.hivemind_utils import HivemindNode
from hivemind_exp.name_utils import get_name_from_peer_id
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer

# 创建日志记录器
logger = logging.getLogger(__name__)

@dataclass
class GRPOArguments:
    # Hivemind参数
    initial_peers: list[str] = field(default_factory=list)  # 初始对等节点列表，用于加入现有的DHT网络
    public_maddr: str | None = None  # 公共多地址，用于向其他节点宣告自己的存在
    host_maddr: str | None = None  # 主机多地址，用于监听连接
    identity_path: str | None = None  # 身份文件路径，用于持久化节点身份
    max_rounds: int = 100  # 最大训练轮次数

    # 模型参数
    dataset_id_or_path: str = "openai/gsm8k"  # 数据集ID或路径
    dataset_splits: str = "train"  # 数据集分割
    tokenizer_name_or_path: str | None = None  # 分词器名称或路径
    number_of_data_samples: int = 50000  # 数据样本数量
    public_maddr: str | None = None  # 公共多地址（重复定义）

    # Hugging Face Hub参数
    hf_token: str | None = None  # Hugging Face的访问令牌


class GRPORunner:
    def get_model(self, args: GRPOConfig, model_name: str):
        """获取预训练的因果语言模型
        
        参数:
            args: GRPO配置
            model_name: 模型名称或路径
            
        返回:
            加载的模型实例
        """
        model_init_kwargs = args.model_init_kwargs or {}
        # 如果启用了梯度检查点，则禁用缓存（不支持）
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        return AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs)

    def get_tokenizer_name(self, model_args: ModelConfig, script_args: GRPOArguments):
        """获取分词器名称
        
        参数:
            model_args: 模型配置
            script_args: GRPO参数
            
        返回:
            分词器名称或路径
            
        异常:
            ValueError: 如果无法解析分词器名称
        """
        if script_args.tokenizer_name_or_path:
            return script_args.tokenizer_name_or_path
        if model_args.model_name_or_path:
            return model_args.model_name_or_path
        raise ValueError("无法解析分词器名称")

    def _dht_kwargs(self, grpo_args):
        """
        构建DHT初始化所需的关键字参数
        
        参数:
            grpo_args: GRPO参数
            
        返回:
            包含DHT初始化参数的字典
        """
        kwargs = {}
        initial_peers = grpo_args.initial_peers
        if initial_peers:
            # 设置初始对等节点，用于加入现有的DHT网络
            kwargs["initial_peers"] = initial_peers

        if public_maddr := grpo_args.public_maddr:
            # 设置公告地址，用于向其他节点宣告自己的存在
            kwargs["announce_maddrs"] = [public_maddr]

        if host_maddr := grpo_args.host_maddr:
            # 设置主机地址，用于监听连接
            kwargs["host_maddrs"] = [host_maddr]

        if identity_path := grpo_args.identity_path:
            # 设置身份文件路径，用于持久化节点身份
            kwargs["identity_path"] = identity_path

        return kwargs

    def _get_animal_name(self, peer_id):
        """
        根据节点ID生成友好的动物名称
        
        参数:
            peer_id: 节点ID
            
        返回:
            生成的动物名称
        """
        animal_name = get_name_from_peer_id(peer_id)
        logger.info(f"🐱 你好 🐈 [{animal_name}] 🦮 [{peer_id}]!")
        return animal_name

    def setup_dht(self, grpo_args):
        """
        设置并启动DHT网络连接
        
        参数:
            grpo_args: GRPO参数，包含初始对等节点等网络配置
            
        返回:
            初始化并启动的DHT实例
        """
        initial_peers = grpo_args.initial_peers
        # 创建并启动DHT实例，使用_dht_kwargs方法构建参数
        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        
        # 根据是否有初始对等节点，决定是加入现有网络还是创建新网络
        if initial_peers:
            # 有初始对等节点，加入现有的蜂群网络
            logger.info(f"🐝 正在加入蜂群网络，初始对等节点 = {initial_peers}")
        else:
            # 没有初始对等节点，创建新的蜂群网络
            first_visible = str(dht.get_visible_maddrs()[0])
            logger.info(f"🤖 正在启动蜂群网络，地址为 {first_visible}")

        # 根据节点ID生成友好的动物名称
        self.name = self._get_animal_name(str(dht.peer_id))
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
        trainer_factory_fn: Callable = HivemindGRPOTrainer,
    ):
        """
        运行GRPO训练流程的主方法
        
        参数:
            model_args: 模型配置参数
            grpo_args: GRPO算法参数
            training_args: 训练配置参数
            initial_datasets_fn: 获取初始数据集的函数
            trainer_factory_fn: 创建训练器的工厂函数
        """
        #########################
        # 记录参数
        #########################
        logger.debug(f"模型参数 {model_args}")
        logger.debug(f"训练/评估参数 {training_args}")

        # 设置批处理大小
        batch_size = 2
        training_args.per_device_train_batch_size = batch_size
        training_args.num_generations = batch_size

        ############################
        # 如果需要，登录Hugging Face Hub
        ############################
        if (grpo_args.hf_token not in [None, "None"]):
            training_args.push_to_hub_token = grpo_args.hf_token
            login(token=training_args.push_to_hub_token, add_to_git_credential=True)
        else:
            training_args.push_to_hub_token = None

        ################
        # 加载分词器
        ################
        tokenizer = AutoTokenizer.from_pretrained(
            self.get_tokenizer_name(model_args, grpo_args),
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        # 如果没有填充标记，使用结束标记作为填充标记
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        #########################
        # 通过Hivemind创建DHT网络
        #########################
        dht = self.setup_dht(grpo_args)

        #####################################
        # 加载数据集，准备和格式化
        #####################################
        train_dataset, test_dataset = initial_datasets_fn()

        #########################
        # 实例化GRPO训练器
        #########################
        model_name_or_path = model_args.model_name_or_path
        assert model_name_or_path
        # 加载模型
        model = self.get_model(training_args, model_name_or_path)

        # 根据是否有初始对等节点，决定是创建普通节点还是协调者节点
        initial_peers = grpo_args.initial_peers
        if initial_peers:
            # 有初始对等节点，创建普通节点
            node = HivemindNode(model_name_or_path, str(dht.peer_id))
        else:
            # 没有初始对等节点，创建协调者节点
            node = HivemindNode.coordinator(model_name_or_path, str(dht.peer_id))

        # 创建训练阶段数据
        stage_data = gsm8k_stage_data(dht, node, train_dataset, test_dataset)
        stage_data.max_rounds = grpo_args.max_rounds
        
        # 创建训练器
        trainer = trainer_factory_fn(
            dht=dht,  # DHT网络实例
            node=node,  # Hivemind节点
            model=model,  # 模型
            tokenizer=tokenizer,  # 分词器
            config=training_args,  # 训练配置
            stage_data=stage_data,  # 训练阶段数据
            log_tag=self.name,  # 日志标签
        )

        ###############
        # 训练循环
        ###############
        logger.info(
            f"开始训练 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，共 {training_args.num_train_epochs} 个训练周期"
        )
        # 开始训练
        trainer.train()
