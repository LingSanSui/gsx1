#!/bin/bash
#rm -rf /root/.cache
# 通用参数设置 (General args)
ROOT=$PWD  # 设置根目录为当前工作目录

# 导出环境变量
export PUB_MULTI_ADDRS      # 公共多地址
export PEER_MULTI_ADDRS     # 对等节点多地址
export HOST_MULTI_ADDRS     # 主机多地址
export IDENTITY_PATH        # 身份路径
export CONNECT_TO_TESTNET   # 是否连接到测试网
export ORG_ID               # 组织ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # HuggingFace下载超时时间（2分钟）

# 检查是否提供了公共多地址，否则设置为默认值
DEFAULT_PUB_MULTI_ADDRS=""  # 默认公共多地址为空
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

# 检查是否提供了对等节点多地址，否则设置为默认值
DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn协调器节点
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

# 检查是否提供了主机多地址，否则设置为默认值
DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"  # 默认监听所有网络接口的38331端口
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# RSA私钥的路径。如果此路径不存在，将创建一个新的密钥对。
# 如果您想要一个新的PeerID，请删除此文件。
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem  # 默认身份文件路径
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

# 询问用户是否连接到测试网
while true; do
    CONNECT_TO_TESTNET=True
    break
    read -p "您想连接到测试网吗？[Y/n] " yn
    yn=${yn:-Y}  # 如果用户直接按回车，默认为"Y"
    case $yn in
        [Yy]* ) CONNECT_TO_TESTNET=True && break;;
        [Nn]* ) CONNECT_TO_TESTNET=False && break;;
        * ) echo ">>> 请回答yes或no。";;
    esac
done

# 如果用户选择连接到测试网
if [ "$CONNECT_TO_TESTNET" = "True" ]; then
    # 运行modal_login服务器
    echo "请登录以创建以太坊服务器钱包"
# 进入modal-login目录
cd modal-login

# 检查npm命令是否存在；如果不存在，则安装npm
source ~/.bashrc

if ! command -v npm >/dev/null 2>&1; then
    # 检测Ubuntu（包括WSL Ubuntu）并相应地安装npm
    if grep -qi "ubuntu" /etc/os-release 2>/dev/null || uname -r | grep -qi "microsoft"; then
        echo "检测到Ubuntu或WSL Ubuntu。通过apt安装npm..."
        sudo apt update
        sudo apt install -y npm
    else
        echo "npm未安装。正在安装npm..."
        curl -L https://npmjs.org/install.sh | sh
    fi
fi

# 使用npm安装依赖项，使用--legacy-peer-deps标志
npm install --legacy-peer-deps # 为什么使用这个标志？因为发现了很多对等依赖问题，这可能是遇到开发错误的原因（https://github.com/gensyn-ai/rl-swarm/issues/74），决定使用npm并用此标志忽略这些问题

# 在后台运行服务器并抑制输出
npm run dev > /dev/null 2>&1 &


    SERVER_PID=$!  # 存储进程ID
    sleep 5
    open http://localhost:3000  # 打开浏览器访问登录页面
    cd ..  # 返回上一级目录

    # 等待modal-login/temp-data/userData.json文件创建
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        echo "等待userData.json文件创建中..."
        sleep 5  # 每5秒检查一次
    done
    echo "找到userData.json文件。继续进行..."

    # 从userData.json文件中提取组织ID
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo "ORG_ID设置为: $ORG_ID"

    # 等待客户端激活API密钥
    echo "等待API密钥激活中..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo "API密钥已激活！继续进行..."
            break
        else
            echo "等待API密钥激活中..."
            sleep 5
        fi
    done

    # 清理服务器进程的函数
    cleanup() {
        echo "正在关闭服务器..."
        kill $SERVER_PID
        #rm -r modal-login/temp-data/*.json
        exit 0
    }

    # 设置陷阱以捕获Ctrl+C并调用cleanup函数
    trap cleanup INT
fi
# 开始执行主要流程！
echo "正在获取依赖项..."
# 安装必要的依赖项
pip install -r "$ROOT"/requirements-hivemind.txt > /dev/null
pip install -r "$ROOT"/requirements.txt > /dev/null

# 检测GPU并选择相应的配置文件
if ! which nvidia-smi; then
   # 没有NVIDIA GPU
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
elif [ -n "$CPU_ONLY" ]; then
   # 或者我们不想使用GPU
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
else
   # 发现NVIDIA GPU
   pip install -r "$ROOT"/requirements_gpu.txt > /dev/null
   CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
fi

echo ">> 完成！"
echo ""
echo ""

# 检查HuggingFace令牌
if [ -n "${HF_TOKEN}" ]; then # 检查HF_TOKEN是否已设置，如果已设置则使用它。否则提示用户选择。
   HUGGINGFACE_ACCESS_TOKEN=${HF_TOKEN}
else
   read -p "您想将在RL swarm中训练的模型推送到Hugging Face Hub吗？[y/N] " yn
   yn=${yn:-N}  # 如果用户直接按回车，默认为"N"
   case $yn in
      [Yy]* ) read -p "请输入您的Hugging Face访问令牌: " HUGGINGFACE_ACCESS_TOKEN;;
      [Nn]* ) HUGGINGFACE_ACCESS_TOKEN="None";;
      * ) echo ">>> 未给出答案，因此不会将模型推送到Hugging Face Hub" && HUGGINGFACE_ACCESS_TOKEN="None";;
   esac
fi

echo ""
echo ""
echo "祝您好运！"

# 根据是否有组织ID选择不同的启动参数
if [ -n "$ORG_ID" ]; then
    # 使用组织ID启动训练（连接到测试网）
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --modal_org_id "$ORG_ID" \
        --config "$CONFIG_PATH"
else
    # 使用P2P参数启动训练（不连接到测试网）
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --public_maddr "$PUB_MULTI_ADDRS" \
        --initial_peers "$PEER_MULTI_ADDRS"\
        --host_maddr "$HOST_MULTI_ADDRS" \
        --config "$CONFIG_PATH"
fi

wait  # 保持脚本运行直到按下Ctrl+C
