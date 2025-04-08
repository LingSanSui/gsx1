import multiprocessing
import hivemind

from . import server_cache

# 客户端使用的DHT单例
# 在主程序中初始化，并在API处理程序中使用
dht: hivemind.DHT | None = None  # 全局DHT实例
dht_cache: server_cache.Cache | None = None  # DHT缓存实例

def setup_global_dht(initial_peers, coordinator, logger):
    """设置全局DHT实例
    
    参数:
        initial_peers: 初始对等节点列表，用于连接到现有DHT网络
        coordinator: 协调者对象，用于管理轮次和阶段
        logger: 日志记录器
    """
    global dht
    global dht_cache
    # 初始化DHT实例并启动
    dht = hivemind.DHT(start=True, initial_peers=initial_peers)
    # 初始化DHT缓存，用于缓存从DHT网络获取的数据
    dht_cache = server_cache.Cache(dht, coordinator, multiprocessing.Manager(), logger)