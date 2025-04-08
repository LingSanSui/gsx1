import json
import logging
from abc import ABC

import requests
from eth_account import Account
from web3 import Web3

# Alchemy API URL，用于连接Gensyn测试网
ALCHEMY_URL = "https://gensyn-testnet.g.alchemy.com/public"

# Gensyn测试网的链ID
MAINNET_CHAIN_ID = 685685

# 群体协调器合约版本
SWARM_COORDINATOR_VERSION = "0.2"
# 群体协调器合约ABI文件路径
SWARM_COORDINATOR_ABI_JSON = (
    f"hivemind_exp/contracts/SwarmCoordinator_{SWARM_COORDINATOR_VERSION}.json"
)
# 群体协调器合约地址
SWARM_COORDINATOR_CONTRACT = "0x2fC68a233EF9E9509f034DD551FF90A79a0B8F82"

# Modal代理服务URL
MODAL_PROXY_URL = "http://localhost:3000/api/"

# 配置日志记录器
logger = logging.getLogger(__name__)


class SwarmCoordinator(ABC):
    """群体协调器抽象基类，定义与区块链交互的接口"""
    @staticmethod
    def coordinator_contract(web3: Web3):
        """创建群体协调器合约实例
        
        Args:
            web3: Web3实例，用于与区块链交互
            
        Returns:
            合约实例
        """
        with open(SWARM_COORDINATOR_ABI_JSON, "r") as f:
            contract_abi = json.load(f)["abi"]

        return web3.eth.contract(address=SWARM_COORDINATOR_CONTRACT, abi=contract_abi)

    def __init__(self, web3: Web3, **kwargs) -> None:
        """初始化群体协调器
        
        Args:
            web3: Web3实例，用于与区块链交互
            **kwargs: 额外参数，传递给父类
        """
        self.web3 = web3
        self.contract = SwarmCoordinator.coordinator_contract(web3)
        super().__init__(**kwargs)

    def register_peer(self, peer_id): 
        """注册节点到区块链（抽象方法，需要子类实现）
        
        Args:
            peer_id: 节点ID
        """
        ...

    def submit_winners(self, round_num, winners): 
        """提交获胜者到区块链（抽象方法，需要子类实现）
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        ...

    def get_bootnodes(self):
        """从区块链获取引导节点列表
        
        Returns:
            引导节点列表
        """
        return self.contract.functions.getBootnodes().call()

    def get_round_and_stage(self):
        """从区块链获取当前轮次和阶段
        
        Returns:
            (轮次编号, 阶段编号)的元组
        """
        with self.web3.batch_requests() as batch:
            batch.add(self.contract.functions.currentRound())
            batch.add(self.contract.functions.currentStage())
            round_num, stage_num = batch.execute()

        return round_num, stage_num


class WalletSwarmCoordinator(SwarmCoordinator):
    """使用钱包私钥直接与区块链交互的群体协调器实现"""
    def __init__(self, private_key: str, **kwargs) -> None:
        """初始化钱包群体协调器
        
        Args:
            private_key: 钱包私钥
            **kwargs: 额外参数，传递给父类
        """
        super().__init__(**kwargs)
        self.account = setup_account(self.web3, private_key)

    def _default_gas(self):
        """获取默认的gas设置
        
        Returns:
            包含gas和gasPrice的字典
        """
        return {
            "gas": 2000000,
            "gasPrice": self.web3.to_wei("1", "gwei"),
        }

    def register_peer(self, peer_id):
        """注册节点到区块链
        
        Args:
            peer_id: 节点ID
        """
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.registerPeer(peer_id).build_transaction(
                self._default_gas()
            ),
        )

    def submit_winners(self, round_num, winners):
        """提交获胜者到区块链
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.submitWinners(
                round_num, winners
            ).build_transaction(self._default_gas()),
        )


class ModalSwarmCoordinator(SwarmCoordinator):
    """通过Modal代理服务与区块链交互的群体协调器实现"""
    def __init__(self, org_id: str, **kwargs) -> None:
        """初始化Modal群体协调器
        
        Args:
            org_id: 组织ID
            **kwargs: 额外参数，传递给父类
        """
        self.org_id = org_id
        super().__init__(**kwargs)

    def register_peer(self, peer_id):
        """通过Modal代理服务注册节点到区块链
        
        Args:
            peer_id: 节点ID
        """
        try:
            send_via_api(self.org_id, "register-peer", {"peerId": peer_id})
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 500:
                raise

            logger.info("调用register-peer端点时发生未知错误！继续执行。")
            # TODO: 验证实际合约错误。
            # logger.info(f"节点ID [{peer_id}] 已经注册！继续执行。")

    def submit_winners(self, round_num, winners):
        """通过Modal代理服务提交获胜者到区块链
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        try:
            args = (
                self.org_id,
                "submit-winner",
                {"roundNumber": round_num, "winners": winners},
            )
            for _ in range(3):
                send_via_api(
                    *args
                )
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 500:
                raise

            logger.info("调用submit-winner端点时发生未知错误！继续执行。")
            # TODO: 验证实际合约错误。
            # logger.info("本轮获胜者已提交！继续执行。")


def send_via_api(org_id, method, args):
    """通过API发送请求到Modal代理服务
    
    Args:
        org_id: 组织ID
        method: API方法名
        args: API参数
        
    Returns:
        API响应的JSON数据
    """
    # 构造URL和负载
    url = MODAL_PROXY_URL + method
    payload = {"orgId": org_id} | args

    # 发送POST请求
    response = requests.post(url, json=payload)
    response.raise_for_status()  # 对HTTP错误抛出异常
    return response.json()


def setup_web3() -> Web3:
    """设置Web3连接到Gensyn测试网
    
    Returns:
        Web3实例
        
    Raises:
        Exception: 连接失败时抛出异常
    """
    # 检查测试网连接
    web3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
    if web3.is_connected():
        logger.info("✅ 已连接到Gensyn测试网")
    else:
        raise Exception("连接到Gensyn测试网失败")
    return web3


def setup_account(web3: Web3, private_key) -> Account:
    """设置账户并检查余额
    
    Args:
        web3: Web3实例
        private_key: 钱包私钥
        
    Returns:
        Account实例
    """
    # 检查钱包余额
    account = web3.eth.account.from_key(private_key)
    balance = web3.eth.get_balance(account.address)
    eth_balance = web3.from_wei(balance, "ether")
    logger.info(f"💰 钱包余额: {eth_balance} ETH")
    return account


def send_chain_txn(
    web3: Web3, account: Account, txn_factory, chain_id=MAINNET_CHAIN_ID
):
    """发送区块链交易
    
    Args:
        web3: Web3实例
        account: 账户实例
        txn_factory: 交易工厂函数，返回交易对象
        chain_id: 链ID，默认为MAINNET_CHAIN_ID
    """
    checksummed = Web3.to_checksum_address(account.address)
    txn = txn_factory() | {
        "chainId": chain_id,
        "nonce": web3.eth.get_transaction_count(checksummed),
    }

    # 签名交易
    signed_txn = web3.eth.account.sign_transaction(txn, private_key=account.key)

    # 发送交易
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    logger.info(f"已发送交易，哈希值: {web3.to_hex(tx_hash)}")
