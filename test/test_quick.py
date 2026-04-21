"""
【测试文件】快速功能验证

用途：快速验证 Agent 的核心功能（工具调用、消息历史）
状态：✅ 可用（用于开发时快速检查代码是否正常）
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from nutritionist_agent import build_nutritionist_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("="*60)
print("快速功能测试")
print("="*60)

agent = build_nutritionist_agent()

# 测试1：简单工具调用
print("\n【测试1】工具调用")
print("-"*60)
config = {"configurable": {"thread_id": "quick_test_001"}}

state = {
    "messages": [HumanMessage(content="我吃了鸡胸肉200大卡，蔬菜50大卡，总共多少？")],
    "image_analyzed": False,
    "image_info": {}
}

result = agent.invoke(state, config=config)

print("\n消息历史:")
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            print(f"  [{i}] AI: 调用工具 {[tc['name'] for tc in msg.tool_calls]}")
        else:
            print(f"  [{i}] AI: {msg.content[:80]}...")
    elif hasattr(msg, 'content'):
        content_preview = str(msg.content)[:80]
        print(f"  [{i}] {msg_type}: {content_preview}...")

print("\n✅ 测试完成！")
