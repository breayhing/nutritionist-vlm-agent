"""
【测试文件】自定义视觉分析指令测试

用途：测试 Agent 能否根据不同需求自定义分析指令
状态：✅ 可用（已验证模型能理解并构建自定义指令）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # 添加父目录到路径
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from nutritionist_agent import build_nutritionist_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("="*80)
print("🧪 测试视觉工具的自定义分析指令")
print("="*80)

agent = build_nutritionist_agent()
config = {"configurable": {"thread_id": "test_custom_instruction_001"}}

image_path = os.path.join(os.path.dirname(__file__), '..', 'test.jpg')  # 图片在父目录
if not os.path.exists(image_path):
    print(f"❌ 测试图片不存在: {image_path}")
    exit(1)

# 测试1：使用默认指令（标准营养分析）
print("\n\n【测试1】默认指令 - 标准营养分析")
print("="*80)
print(f"用户：请分析这张图片：{image_path}")
print("-"*80)

state1 = {
    "messages": [HumanMessage(content=f"请分析这张图片：{image_path}")],
    "image_analyzed": False,
    "image_info": {}
}

result1 = agent.invoke(state1, config=config)
print("\n消息历史:")
for i, msg in enumerate(result1["messages"]):
    msg_type = type(msg).__name__
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [{i}] AI 调用: {tc['name']}")
                if 'args' in tc:
                    print(f"       参数: {list(tc['args'].keys())}")
        else:
            print(f"  [{i}] AI: {msg.content[:100]}...")
    elif hasattr(msg, 'content'):
        content_preview = str(msg.content)[:100]
        print(f"  [{i}] {msg_type}: {content_preview}...")


# 测试2：自定义指令 - 只识别食材
print("\n\n【测试2】自定义指令 - 只识别食材名称")
print("="*80)
print("用户：帮我看看这张图里有哪些食材，不需要热量信息")
print("-"*80)

# 注意：这里我们期望模型能理解用户需求并自动构建合适的 analysis_instruction
state2 = {
    "messages": [HumanMessage(content="帮我看看这张图里有哪些食材，不需要热量信息")],
    "image_analyzed": False,
    "image_info": {}
}

# 由于模型不知道图片路径，它应该会询问
result2 = agent.invoke(state2, config=config)
print("\nAI 回复:")
for msg in result2["messages"]:
    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        print(msg.content)


# 测试3：直接提供图片和自定义指令
print("\n\n【测试3】直接提供图片和特定分析需求")
print("="*80)
print(f"用户：分析 {image_path}，重点看蛋白质含量够不够")
print("-"*80)

state3 = {
    "messages": [HumanMessage(content=f"分析 {image_path}，重点看蛋白质含量够不够")],
    "image_analyzed": False,
    "image_info": {}
}

result3 = agent.invoke(state3, config=config)
print("\n消息历史:")
for i, msg in enumerate(result3["messages"]):
    msg_type = type(msg).__name__
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [{i}] AI 调用: {tc['name']}")
                if 'args' in tc and 'analysis_instruction' in tc['args']:
                    instruction = tc['args']['analysis_instruction']
                    print(f"       分析指令: {instruction[:80]}...")
        else:
            print(f"  [{i}] AI: {msg.content[:100]}...")
    elif hasattr(msg, 'content'):
        content_preview = str(msg.content)[:100]
        print(f"  [{i}] {msg_type}: {content_preview}...")

print("\n" + "="*80)
print("✅ 测试完成！")
print("="*80)
print("\n💡 观察要点：")
print("  1. 测试1应该使用默认的 analysis_instruction")
print("  2. 测试2中模型应该询问图片路径")
print("  3. 测试3中模型应该根据'重点看蛋白质'构建自定义指令")
print("="*80)
