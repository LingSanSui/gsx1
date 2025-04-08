import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from threading import Thread

import aiofiles
from hivemind_exp.chain_utils import ModalSwarmCoordinator, setup_web3
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json

# 导入DHT工具函数和名称工具函数
from hivemind_exp.dht_utils import *
from hivemind_exp.name_utils import *

# 导入全局DHT实例
from . import global_dht

# UI从文件系统提供服务
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "ui", "dist")

# 缓存index.html内容
index_html = None


async def load_index_html():
    """异步加载index.html文件内容到内存"""
    global index_html
    if index_html is None:
        index_path = os.path.join(BASE_DIR, "ui", "dist", "index.html")
        async with aiofiles.open(index_path, mode="r") as f:
            index_html = await f.read()


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI()
# 从环境变量获取端口号，默认为8000
port = os.getenv("SWARM_UI_PORT", "8000")

try:
    port = int(port)
except ValueError:
    logger.warning(f"invalid port {port}. Defaulting to 8000")
    port = 8000

# 配置uvicorn服务器
config = uvicorn.Config(
    app,
    host="0.0.0.0",
    port=port,
    timeout_keep_alive=10,
    timeout_graceful_shutdown=10,
    h11_max_incomplete_event_size=8192,  # 最大请求头大小（字节）
)

# 创建uvicorn服务器实例
server = uvicorn.Server(config)


@app.exception_handler(Exception)
async def internal_server_error_handler(request: Request, exc: Exception):
    """全局异常处理器，捕获所有未处理的异常并返回500错误"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "message": str(exc),
        },
    )


@app.get("/api/healthz")
async def get_health():
    """健康检查API，检查DHT是否正常运行"""
    # 获取上次轮询DHT的时间
    lpt = global_dht.dht_cache.get_last_polled()
    if lpt is None:
        # 如果从未轮询过DHT，返回500错误
        raise HTTPException(status_code=500, detail="dht never polled")

    # 计算上次轮询到现在的时间差
    diff = datetime.now() - lpt
    if diff > timedelta(minutes=5):
        # 如果超过5分钟未轮询，返回500错误
        raise HTTPException(status_code=500, detail="dht last poll exceeded 5 minutes")

    # 返回健康状态和上次轮询时间
    return {
        "message": "OK",
        "lastPolled": diff,
    }


@app.get("/api/round_and_stage")
def get_round_and_stage():
    """获取当前轮次和阶段的API"""
    # 从DHT缓存获取当前轮次和阶段
    r, s = global_dht.dht_cache.get_round_and_stage()

    # 返回轮次和阶段信息
    return {
        "round": r,
        "stage": s,
    }


@app.get("/api/leaderboard")
def get_leaderboard():
    """获取排行榜数据的API"""
    # 从DHT缓存获取排行榜数据
    leaderboard = global_dht.dht_cache.get_leaderboard()
    res = dict(leaderboard)

    # 如果排行榜数据存在，返回领导者列表和总数
    if res is not None:
        return {
            "leaders": res.get("leaders", []),
            "total": res.get("total", 0),
        }


@app.get("/api/rewards-history")
def get_rewards_history():
    """获取奖励历史记录的API"""
    # 从DHT缓存获取排行榜数据
    leaderboard = global_dht.dht_cache.get_leaderboard()
    res = dict(leaderboard)

    # 如果排行榜数据存在，返回奖励历史记录
    if res is not None:
        return {
            "leaders": res.get("rewardsHistory", []),
        }


@app.get("/api/name-to-id")
def get_id_from_name(name: str = Query("")):
	"""根据节点名称获取节点ID的API"""
	# 从DHT缓存获取排行榜数据
	leaderboard = global_dht.dht_cache.get_leaderboard()
	# 提取所有领导者的ID
	leader_ids = [leader["id"] for leader in leaderboard["leaders"]] or []

	# 根据名称搜索对应的节点ID
	peer_id = search_peer_ids_for_name(leader_ids, name)
	return {
		"id": peer_id,
	}

@app.post("/api/id-to-name")
async def id_to_name(request: Request):
    """根据节点ID列表获取节点名称的API"""
    # 检查请求体大小（限制为100KB）
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 100 * 1024:  # 100KB（字节）
        raise HTTPException(
            status_code=413,
            detail="Request body too large. Maximum size is 100KB."
        )

    # 解析请求体
    try:
        body = await request.json()
        if not isinstance(body, list):
            raise HTTPException(
                status_code=400,
                detail="Request body must be a list of peer IDs"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request body: {str(e)}"
        )

    # 验证输入大小
    if len(body) > 1000:  # 限制可处理的ID数量
        raise HTTPException(
            status_code=400,
            detail="Too many peer IDs. Maximum is 1000."
        )

    # 处理每个ID
    id_to_name_map = {}
    for peer_id in body:
        try:
            name = get_name_from_peer_id(peer_id)
            if name is not None:
                id_to_name_map[peer_id] = name
        except Exception as e:
            logger.error(f"Error looking up name for peer ID {peer_id}: {str(e)}")

    return id_to_name_map

@app.get("/api/gossip")
def get_gossip():
    """获取节点间消息（gossip）的API"""
    # 从DHT缓存获取gossip消息
    gs = global_dht.dht_cache.get_gossips()
    return dict(gs)


if os.getenv("API_ENV") != "dev":
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(DIST_DIR, "assets")),
        name="assets",
    )
    app.mount(
        "/fonts", StaticFiles(directory=os.path.join(DIST_DIR, "fonts")), name="fonts"
    )
    app.mount(
        "/images",
        StaticFiles(directory=os.path.join(DIST_DIR, "images")),
        name="images",
    )


@app.get("/{full_path:path}")
async def catch_all(full_path: str, request: Request):
    # Development reverse proxies to ui dev server
    if os.getenv("API_ENV") == "dev":
        logger.info(
            f"proxying {full_path} into local UI development environment on 5173..."
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url=f"http://localhost:5173/{full_path}", headers=request.headers
            )
            headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ["content-length", "transfer-encoding"]
            }
            return Response(
                content=resp.content, status_code=resp.status_code, headers=headers
            )

    # Live environment (serve from dist)
    # We don't want to cache index.html, but other static assets are fine to cache.
    await load_index_html()
    return HTMLResponse(
        content=index_html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip", "--initial_peers", help="initial peers", nargs="+", type=str, default=[]
    )
    return parser.parse_args()


def populate_cache():
    logger.info("populate_cache initialized")
    try:
        while True:
            logger.info("pulling latest dht data...")
            global_dht.dht_cache.poll_dht()
            time.sleep(10)
            logger.info("dht polled")
    except Exception as e:
        logger.error("uncaught exception while polling dht", e)


def main(args):
    coordinator = ModalSwarmCoordinator("", web3=setup_web3()) # Only allows contract calls
    initial_peers = coordinator.get_bootnodes()

    # Supplied with the bootstrap node, the client will have access to the DHT.
    logger.info(f"initializing DHT with peers {initial_peers}")
    global_dht.setup_global_dht(initial_peers, coordinator, logger)

    thread = Thread(target=populate_cache)
    thread.daemon = True
    thread.start()

    logger.info(f"initializing server on port {port}")
    server.run()


if __name__ == "__main__":
    main(parse_arguments())
