import re

# 标签模式模板，用于从文本中提取被特定标签包围的内容
TAGGED_PATTERN_TEMPLATE = r"<{0}>\n*(.*?)\n*</{0}>"

def _extract_tagged(text, tag):
    """从文本中提取被特定标签包围的内容
    
    参数:
        text: 包含标签的文本
        tag: 标签名称
        
    返回:
        提取出的内容
    """
    matches = re.findall(TAGGED_PATTERN_TEMPLATE.format(tag), text)
    return matches[0]

def stage1_message(node_key: str, question:str, ts, outputs: dict ):
    """生成第一阶段的消息格式
    
    参数:
        node_key: 节点键
        question: 问题
        ts: 时间戳
        outputs: 输出字典
        
    返回:
        格式化的消息字符串
    """
    answer = outputs['answer']
    return f"{question}...Answer: {answer}"

def stage2_message(node_key: str, question:str, ts, outputs: dict ):
    """生成第二阶段的消息格式
    
    参数:
        node_key: 节点键
        question: 问题
        ts: 时间戳
        outputs: 输出字典
        
    返回:
        格式化的消息字符串
    """
    try:
        opinion = outputs["agent_opinion"][node_key]
        explain = _extract_tagged(opinion, "explain").strip()
        identify = _extract_tagged(opinion, "identify").strip()
        return f"{explain}...Identify: {identify}"
    except (ValueError, KeyError, IndexError):
        # 如果无法提取第二阶段的消息，则回退到第一阶段的消息格式
        return stage1_message(node_key, question, ts, outputs)

def stage3_message(node_key: str, question:str, ts, outputs: dict ):
    """生成第三阶段的消息格式
    
    参数:
        node_key: 节点键
        question: 问题
        ts: 时间戳
        outputs: 输出字典
        
    返回:
        格式化的消息字符串
    """
    try:
        decision = outputs["final_agent_decision"][node_key]
        summarize_feedback = _extract_tagged(decision, "summarize_feedback").strip()
        majority =_extract_tagged(decision, "majority").strip()
        return f"{summarize_feedback}...Majority: {majority}"
    except (ValueError, KeyError, IndexError):
        # 如果无法提取第三阶段的消息，则回退到第一阶段的消息格式
        return stage1_message(node_key, question, ts, outputs)
