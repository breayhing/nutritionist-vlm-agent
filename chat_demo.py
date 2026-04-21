"""
================================================================================
【推荐使用】交互式营养师助手 - 日常使用入口
================================================================================

用途：这是给用户使用的交互式对话程序，可以连续多轮对话

如何使用：
1. 准备你的食物图片，放到 project/ 目录（例如：my_lunch.jpg）
2. 运行：python chat_demo.py
3. 输入问题，例如：
   - "分析 my_lunch.jpg"
   - "这个热量算高吗？"
   - "帮我看看 dinner.png 的营养"
4. 输入 'quit' 或 'exit' 退出

特点：
- ✅ 支持连续对话（有记忆）
- ✅ 自动从对话中提取图片路径
- ✅ 可以随时发送新图片
- ✅ 不需要修改代码

注意：
- 这是正式的使用程序，不是测试文件
- 图片文件不要放在 test/ 文件夹，要放在 project/ 目录
================================================================================
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from nutritionist_agent import build_nutritionist_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("="*80)
print("🥗 交互式营养师助手")
print("="*80)
print("\n💡 使用说明：")
print("  - 直接输入文字进行对话")
print("  - 提到图片时，写上文件路径，例如：'分析 my_food.jpg'")
print("  - 输入 'quit' 或 'exit' 退出")
print("="*80)

# 创建 Agent
agent = build_nutritionist_agent()
thread_id = "interactive_chat_001"
config = {"configurable": {"thread_id": thread_id}}

# 初始化状态
current_state = {
    "messages": [],
    "image_analyzed": False,
    "image_info": {}
}

print("\n👋 你好！我是你的营养师助手，有什么可以帮助你的？\n")

while True:
    # 获取用户输入
    try:
        user_input = input("👤 你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 再见！")
        break
    
    if not user_input:
        continue
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 再见！")
        break
    
    # 构建消息
    current_state["messages"] = [HumanMessage(content=user_input)]
    
    print("\n🤖 AI 思考中...")
    print("-"*80)
    
    try:
        # 运行 Agent
        result = agent.invoke(current_state, config=config)
        
        # 显示 AI 回复（只显示最新的 AI 消息）
        # 从后往前找，找到第一条符合条件的 AI 消息
        latest_ai_message = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                latest_ai_message = msg
                break
        
        if latest_ai_message:
            print(f"\n🤖 AI：{latest_ai_message.content}")
        else:
            print("\n🤖 AI：（无回复）")
        
        # 更新状态（保留历史）
        current_state = result
        
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ 错误：{error_str}")
        
        # 如果是超时错误，给出更友好的提示
        if "timeout" in error_str.lower() or "timed out" in error_str.lower():
            print("\n💡 提示：API 请求超时，可能原因：")
            print("   1. 网络连接不稳定")
            print("   2. API 服务器繁忙")
            print("   3. 图片太大，处理时间长")
            print("\n建议：")
            print("   - 检查网络连接")
            print("   - 稍后重试")
            print("   - 如果使用图片，尝试压缩后再上传")
        else:
            print("请重试或检查 API 配置")
    
    print("\n" + "-"*80)
