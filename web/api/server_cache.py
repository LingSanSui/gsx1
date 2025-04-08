import hashlib
import itertools
from datetime import datetime
from typing import Sequence

from .gossip_utils import *

from hivemind_exp.dht_utils import *
from hivemind_exp.name_utils import get_name_from_peer_id
from .gossip_utils import stage1_message, stage2_message, stage3_message


class Cache:
    """DHT数据缓存类，用于缓存从DHT网络获取的数据
    
    该类负责定期从DHT网络获取数据并缓存在内存中，减少对DHT的直接访问，提高性能。
    """
    def __init__(self, dht, coordinator, manager, logger):
        """初始化缓存
        
        参数:
            dht: DHT实例，用于与DHT网络通信
            coordinator: 协调者实例，用于获取轮次和阶段信息
            manager: 多进程管理器，用于创建共享数据结构
            logger: 日志记录器
        """
        self.dht = dht  # DHT实例
        self.coordinator = coordinator  # 协调者实例

        self.manager = manager  # 多进程管理器
        self.logger = logger  # 日志记录器

        self.lock = manager.Lock()  # 用于线程安全的锁
        self.reset()  # 重置缓存

    def reset(self):
        """重置缓存数据
        
        初始化所有共享数据结构，清除现有缓存
        """
        self.leaderboard = self.manager.dict()  # 排行榜缓存
        self.rewards_history = self.manager.dict()  # 奖励历史记录缓存
        self.gossips = self.manager.dict()  # 节点间消息缓存

        self.current_round = self.manager.Value("i", -1)  # 当前轮次，初始为-1
        self.current_stage = self.manager.Value("i", -1)  # 当前阶段，初始为-1

        self.last_polled = None  # 上次轮询DHT的时间

    def get_round_and_stage(self):
        """获取当前轮次和阶段
        
        返回:
            当前轮次和阶段的元组
        """
        return self.current_round.value, self.current_stage.value

    def get_leaderboard(self):
        """获取排行榜数据
        
        返回:
            排行榜数据字典
        """
        return dict(self.leaderboard)

    def get_gossips(self, since_round=0):
        """获取节点间消息
        
        参数:
            since_round: 起始轮次，默认为0
            
        返回:
            节点间消息字典
        """
        return dict(self.gossips)

    def get_last_polled(self):
        """获取上次轮询DHT的时间
        
        返回:
            上次轮询时间的datetime对象
        """
        return self.last_polled

    def poll_dht(self):
        """轮询DHT网络并更新缓存
        
        从DHT网络获取最新数据并更新本地缓存，包括轮次和阶段、排行榜和节点间消息
        """
        try:
            self._get_round_and_stage()  # 获取最新的轮次和阶段
            self._get_leaderboard()      # 获取最新的排行榜数据
            self._get_gossip()           # 获取最新的节点间消息

            with self.lock:
                self.last_polled = datetime.now()  # 更新上次轮询时间
        except Exception as e:
            self.logger.error("cache failed to poll dht: %s", e)

    def _get_dht_value(self, **kwargs):
        return get_dht_value(self.dht, beam_size=100, **kwargs)

    def _get_round_and_stage(self):
        try:
            r, s = self.coordinator.get_round_and_stage()
            self.logger.info(f"cache polled round and stage: r={r}, s={s}")
            with self.lock:
                self.current_round.value = r
                self.current_stage.value = s
        except ValueError as e:
            self.logger.warning(
                "could not get current round or stage; default to -1: %s", e
            )

    def _last_round_and_stage(self, round_num, stage):
        r = round_num
        s = stage - 1
        if s == -1:
            s = 2
            r -= 1

        return max(0, r), max(0, s)

    def _get_leaderboard(self):
        try:
            curr_round = self.current_round.value
            curr_stage = self.current_stage.value

            raw = self._get_dht_value(
                key=leaderboard_key(
                    *self._last_round_and_stage(curr_round, curr_stage)
                ),
                latest=True,
            )

            # Create entries for all participants
            all_entries = [
                {
                    "id": str(t[0]),
                    "nickname": get_name_from_peer_id(t[0]),
                    "score": t[1],
                    "values": [],
                }
                for t in (raw or [])
            ]
            self.logger.info(">>> lb_entries length: %d", len(all_entries))

            current_history = []
            with self.lock:
                for entry in all_entries:
                    latestScore = entry["score"]
                    id = entry["id"]
                    nn = entry["nickname"]

                    past_scores = self.rewards_history.get(id, [])
                    next_scores = (
                        past_scores
                        + [{"x": int(datetime.now().timestamp()), "y": latestScore}][
                            -100:
                        ]
                    )
                    self.logger.info(
                        ">>> id: %s, past_scores length: %d, next_scores length: %d",
                        id,
                        len(past_scores),
                        len(next_scores),
                    )
                    self.rewards_history[id] = next_scores
                    current_history.append(
                        {
                            "id": id,
                            "nickname": nn,
                            "values": next_scores,
                        }
                    )

            with self.lock:
                self.leaderboard = {
                    "leaders": all_entries,
                    "total": len(raw) if raw else 0,
                    "rewardsHistory": current_history,
                }
        except Exception as e:
            self.logger.warning("could not get leaderboard data: %s", e)

    def _get_gossip(self):
        STAGE_GOSSIP_LIMIT = 10  # Most recent.
        STAGE_MESSAGE_FNS = [stage1_message, stage2_message, stage3_message]

        round_gossip = []
        start_time = datetime.now()
        try:
            # Basically a proxy for the reachable peer group.

            curr_round = self.current_round.value
            curr_stage = self.current_stage.value

            prev_rewards = self._get_dht_value(
                key=rewards_key(*self._last_round_and_stage(curr_round, curr_stage)),
                latest=True,
            )
            if not prev_rewards:
                raise ValueError("missing prev_rewards")

            nodes: Sequence[str] = prev_rewards.keys()
            start_round = max(0, curr_round - 3)
            for round_num, stage, node_key in itertools.product(
                range(start_round, curr_round + 1),
                range(0, 3),
                nodes,
            ):
                # Check if we've exceeded 10 seconds
                # Adding this as a stop gap to make sure the gossip collection doesn't stop other data from being polled.
                if (datetime.now() - start_time).total_seconds() > 10:
                    self.logger.warning(">>> gossip collection timed out after 10s")
                    break

                if round_num > curr_round or (
                    round_num == curr_round and stage > curr_stage
                ):
                    break

                key = outputs_key(node_key, round_num, stage)
                if outputs := self._get_dht_value(key=key):
                    sorted_outputs = sorted(
                        list(outputs.items()), key=lambda t: t[1][0]
                    )
                    for question, (ts, outputs) in sorted_outputs[-STAGE_GOSSIP_LIMIT:]:
                        gossip_id = hashlib.md5(
                            f"{node_key}_{round_num}_{stage}_{question}".encode()
                        ).hexdigest()
                        if stage < len(STAGE_MESSAGE_FNS):
                            message = STAGE_MESSAGE_FNS[stage](
                                node_key, question, ts, outputs
                            )
                        else:
                            message = f"Cannot render output for unknown stage {stage}"
                        round_gossip.append(
                            (
                                ts,
                                {
                                    "id": gossip_id,
                                    "message": message,
                                    "node": get_name_from_peer_id(node_key),
                                },
                            )
                        )
        except Exception as e:
            self.logger.warning("could not get gossip: %s", e)
        finally:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                ">>> completed gossip with %d messages in %.2fs",
                len(round_gossip),
                elapsed,
            )

        with self.lock:
            self.gossips = {
                "messages": [msg for _, msg in sorted(round_gossip, reverse=True)]
                or [],
            }
