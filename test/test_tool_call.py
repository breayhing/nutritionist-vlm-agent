"""
【测试文件】基础工具调用测试

用途：测试 calculate_total_calories 工具是否正常工作
状态：✅ 可用（用于快速验证工具调用功能）
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from nutritionist_agent import build_nutritionist_agent

# 加载环境变量
load_dotenv()

print("="*60)
print("🧪 测试工具调用功能")
print("="*60)

# 创建 Agent
agent = build_nutritionist_agent()

# 配置线程 ID
thread_id = "test_tool_call_001"
config = {"configurable": {"thread_id": thread_id}}

# 测试：触发工具调用
print("\n\n📝 测试：热量计算（应该触发工具调用）")
print("-"*60)

state = {
    "messages": [
        HumanMessage(content="我吃了鸡胸肉200大卡，米饭150大卡，蔬菜50大卡，帮我算一下总热量是多少？")
    ],
    "image_analyzed": True,
    "image_info": {}
}

try:
    result = agent.invoke(state, config=config)
    
    print("\n" + "="*60)
    print("📊 最终回复：")
    print("="*60)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
            
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ 测试完成！")
print("="*60)
