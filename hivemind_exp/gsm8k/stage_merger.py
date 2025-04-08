from typing import Any


# 合并第一阶段问题数据
def merge_stage1_question(outputs: dict[str, dict[str, Any]]):
    # TODO: 当前问题+答案在每个文件中都被替换，这很浪费，可以优化
    # TODO: 如果一个代理回答多次（或来自同一代理ID哈希的多个答案），当前实现只会保留循环中最后看到的。是否应该允许多个答案？
    merged = {"question": None, "answer": None, "agent_answers": {}}
    for o in outputs.values():
        merged["question"] = o["question"]
        merged["answer"] = o["answer"]
        merged["agent_answers"].update(o["agent_answers"])
    # 填充默认值。TODO: 决定这是否是一个好选择。
    for agent in outputs:
        if agent not in merged["agent_answers"]:
            merged["agent_answers"].update({agent: "No answer received..."})
    return merged


# 合并第二阶段问题数据
def merge_stage2_question(outputs: dict[str, dict[str, Any]]):
    # TODO: 当前问题+答案在每个文件中都被替换，这很浪费，可以优化
    # TODO: 如果一个代理回答多次（或来自同一代理ID哈希的多个答案），当前实现只会保留循环中最后看到的。是否应该允许多个答案？
    merged = {
        "question": None,
        "answer": None,
        "stage2_prompt": None,
        "agent_opinion": {},
    }
    for o in outputs.values():
        for col in ["question", "answer", "stage2_prompt"]:
            if col in o:
                merged[col] = o[col]
        if "agent_opinion" in o:
            merged["agent_opinion"].update(o["agent_opinion"])
    # 填充默认值。TODO: 决定这是否是一个好选择。
    for agent in outputs:
        if agent not in merged["agent_opinion"]:
            merged["agent_opinion"].update({agent: "No feedback received..."})
    return merged
