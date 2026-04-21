"""
【测试文件】全面功能测试

用途：测试所有功能（纯文本对话、图片分析、多轮记忆、工具调用）
状态：⚠️  需要优化（长对话时可能超时，建议用于完整功能验证）
"""
import os
import sys
# 设置输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from nutritionist_agent import build_nutritionist_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

print("="*80)
print("🧪 全面测试营养师 Agent")
print("="*80)

# 创建 Agent
agent = build_nutritionist_agent()

# ============================================
# 测试1：纯文本多轮对话（测试记忆功能）
# ============================================
print("\n\n" + "="*80)
print("📝 测试1：纯文本多轮对话 - 验证记忆功能")
print("="*80)

thread_id_1 = "test_text_memory_001"
config_1 = {"configurable": {"thread_id": thread_id_1}}

# 第1轮：询问健康饮食原则
print("\n【第1轮】用户：健康饮食的基本原则是什么？")
print("-"*80)
state1 = {
    "messages": [HumanMessage(content="健康饮食的基本原则是什么？")],
    "image_analyzed": False,
    "image_info": {}
}
result1 = agent.invoke(state1, config=config_1)
for msg in result1["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"AI：{msg.content[:300]}...")

# 第2轮：追问减脂期饮食（应该记得上一轮在聊饮食）
print("\n【第2轮】用户：那减脂期应该怎么吃？")
print("-"*80)
state2 = {
    "messages": [HumanMessage(content="那减脂期应该怎么吃？")],
    "image_analyzed": False,
    "image_info": {}
}
result2 = agent.invoke(state2, config=config_1)
for msg in result2["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"AI：{msg.content[:300]}...")

# 第3轮：提供具体数据并请求计算（应该触发工具调用）
print("\n【第3轮】用户：我吃了鸡胸肉200大卡，蔬菜50大卡，总共多少？")
print("-"*80)
state3 = {
    "messages": [HumanMessage(content="我吃了鸡胸肉200大卡，蔬菜50大卡，总共多少？")],
    "image_analyzed": False,
    "image_info": {}
}
result3 = agent.invoke(state3, config=config_1)
print("完整消息历史：")
for i, msg in enumerate(result3["messages"]):
    msg_type = type(msg).__name__
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            print(f"  [{i}] {msg_type}: 调用工具 {msg.tool_calls}")
        else:
            print(f"  [{i}] {msg_type}: {msg.content[:100]}...")
    elif isinstance(msg, ToolMessage):
        print(f"  [{i}] {msg_type}: {msg.content}")
    elif isinstance(msg, HumanMessage):
        content_preview = str(msg.content)[:50]
        print(f"  [{i}] {msg_type}: {content_preview}...")

# 第4轮：基于之前的计算结果追问（测试是否记得工具结果）
print("\n【第4轮】用户：这个热量算高吗？")
print("-"*80)
state4 = {
    "messages": [HumanMessage(content="这个热量算高吗？")],
    "image_analyzed": False,
    "image_info": {}
}
result4 = agent.invoke(state4, config=config_1)
for msg in result4["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"AI：{msg.content[:300]}...")

print("\n✅ 测试1完成：纯文本多轮对话")


# ============================================
# 测试2：带图片的对话（测试视觉工具）
# ============================================
print("\n\n" + "="*80)
print("📷 测试2：带图片的对话 - 验证视觉工具调用")
print("="*80)

thread_id_2 = "test_image_002"
config_2 = {"configurable": {"thread_id": thread_id_2}}

image_path = "../test.jpg"
if os.path.exists(image_path):
    print(f"\n【第1轮】用户：请帮我分析这张图片的热量和营养：{image_path}")
    print("-"*80)
    
    state_img1 = {
        "messages": [HumanMessage(content=f"请帮我分析这张图片的热量和营养：{image_path}")],
        "image_analyzed": False,
        "image_info": {}
    }
    
    result_img1 = agent.invoke(state_img1, config=config_2)
    
    print("完整消息历史：")
    for i, msg in enumerate(result_img1["messages"]):
        msg_type = type(msg).__name__
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"  [{i}] {msg_type}: 调用工具 {[tc['name'] for tc in msg.tool_calls]}")
            else:
                print(f"  [{i}] {msg_type}: {msg.content[:150]}...")
        elif isinstance(msg, ToolMessage):
            content_preview = str(msg.content)[:150]
            print(f"  [{i}] {msg_type}: {content_preview}...")
        elif isinstance(msg, HumanMessage):
            content_preview = str(msg.content)[:80]
            print(f"  [{i}] {msg_type}: {content_preview}...")
    
    # 显示最终回复
    print("\nAI 最终回复：")
    for msg in result_img1["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            print(msg.content)
else:
    print(f"⚠️  跳过测试2：测试图片不存在 ({image_path})")


# ============================================
# 测试3：图片在后续轮次发送（测试灵活性）
# ============================================
print("\n\n" + "="*80)
print("🔄 测试3：图片在后续轮次发送 - 验证灵活性")
print("="*80)

thread_id_3 = "test_image_later_003"
config_3 = {"configurable": {"thread_id": thread_id_3}}

if os.path.exists(image_path):
    # 第1轮：纯文本对话
    print("\n【第1轮】用户：我想了解如何搭配健康的午餐")
    print("-"*80)
    state_later1 = {
        "messages": [HumanMessage(content="我想了解如何搭配健康的午餐")],
        "image_analyzed": False,
        "image_info": {}
    }
    result_later1 = agent.invoke(state_later1, config=config_3)
    for msg in result_later1["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"AI：{msg.content[:200]}...")
    
    # 第2轮：发送图片
    print(f"\n【第2轮】用户：这是我今天的午餐，请分析：{image_path}")
    print("-"*80)
    state_later2 = {
        "messages": [HumanMessage(content=f"这是我今天的午餐，请分析：{image_path}")],
        "image_analyzed": False,
        "image_info": {}
    }
    result_later2 = agent.invoke(state_later2, config=config_3)
    
    print("完整消息历史：")
    for i, msg in enumerate(result_later2["messages"]):
        msg_type = type(msg).__name__
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"  [{i}] {msg_type}: 调用工具 {[tc['name'] for tc in msg.tool_calls]}")
            else:
                print(f"  [{i}] {msg_type}: {msg.content[:150]}...")
        elif isinstance(msg, ToolMessage):
            content_preview = str(msg.content)[:150]
            print(f"  [{i}] {msg_type}: {content_preview}...")
        elif isinstance(msg, HumanMessage):
            content_preview = str(msg.content)[:80]
            print(f"  [{i}] {msg_type}: {content_preview}...")
    
    # 显示最终回复
    print("\nAI 最终回复：")
    for msg in result_later2["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            print(msg.content)
else:
    print(f"⚠️  跳过测试3：测试图片不存在 ({image_path})")


# ============================================
# 测试总结
# ============================================
print("\n\n" + "="*80)
print("✅ 所有测试完成！")
print("="*80)
print("\n📊 测试覆盖：")
print("  ✓ 纯文本多轮对话（4轮）")
print("  ✓ 工具调用（calculate_total_calories）")
print("  ✓ 记忆功能（跨轮次记住上下文）")
if os.path.exists(image_path):
    print("  ✓ 视觉工具调用（analyze_food_image）")
    print("  ✓ 图片在首轮发送")
    print("  ✓ 图片在后续轮次发送")
else:
    print("  ⚠ 图片测试跳过（test.jpg 不存在）")
print("="*80)
