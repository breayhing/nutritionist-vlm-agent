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
- ✅ 支持用户档案持久化
- ✅ 可以通过对话更新个人信息

新增命令：
- 输入 'profile' 或 '档案' 查看当前个人信息
- 输入 'setup' 重新设置个人信息

注意：
- 这是正式的使用程序，不是测试文件
- 图片文件不要放在 test/ 文件夹，要放在 project/ 目录
================================================================================
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from nutritionist_agent import build_nutritionist_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ==============================================
# 用户档案管理
# ==============================================
USER_PROFILE_FILE = "user_profile.json"
DEFAULT_PROFILE = {
    "height_cm": 175,
    "weight_kg": 70,
    "age": 25,
    "gender": "male",
    "activity_level": "moderate",
    "goal": "maintain",
    "allergies": []
}


def load_user_profile():
    """加载用户档案"""
    if os.path.exists(USER_PROFILE_FILE):
        try:
            with open(USER_PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()


def save_user_profile(profile):
    """保存用户档案"""
    try:
        with open(USER_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False


def calculate_bmr(profile):
    """计算基础代谢"""
    weight = profile["weight_kg"]
    height = profile["height_cm"]
    age = profile["age"]
    if profile["gender"] == "male":
        return int(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age))
    else:
        return int(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))


def calculate_tdee(bmr, activity_level):
    """计算每日总消耗"""
    multipliers = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
        "active": 1.725, "very_active": 1.9
    }
    return int(bmr * multipliers.get(activity_level, 1.55))


def get_daily_calorie_goal(profile):
    """计算每日热量目标"""
    bmr = calculate_bmr(profile)
    tdee = calculate_tdee(bmr, profile["activity_level"])
    if profile["goal"] == "fat_loss":
        return int(tdee * 0.8)
    elif profile["goal"] == "muscle_gain":
        return int(tdee * 1.1)
    else:
        return tdee


def get_profile_text(profile):
    """获取用户档案的文本描述"""
    daily_goal = get_daily_calorie_goal(profile)
    gender_text = "男" if profile["gender"] == "male" else "女"
    activity_text = {
        "sedentary": "久坐办公", "light": "轻度活动",
        "moderate": "中度活动", "active": "经常运动",
        "very_active": "高强度运动"
    }.get(profile["activity_level"], profile["activity_level"])
    goal_text = {
        "fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持体重"
    }.get(profile["goal"], profile["goal"])

    return f"""
【用户档案】
- 基本信息: {gender_text}, {profile['age']}岁, 身高{profile['height_cm']}cm, 体重{profile['weight_kg']}kg
- 活动水平: {activity_text}
- 健康目标: {goal_text}
- 每日建议热量: {daily_goal} 千卡
- 过敏原: {', '.join(profile['allergies']) if profile['allergies'] else '无'}
"""


def setup_user_profile():
    """首次设置用户档案"""
    print("\n" + "="*60)
    print("👋 欢迎使用！让我们先设置你的个人信息")
    print("="*60)

    profile = DEFAULT_PROFILE.copy()

    try:
        profile["gender"] = input("性别 (男/女) [默认: 男]: ").strip()
        if profile["gender"] not in ["男", "女", ""]:
            profile["gender"] = "male"
        elif profile["gender"] == "女":
            profile["gender"] = "female"
        else:
            profile["gender"] = "male"

        age_input = input(f"年龄 [默认: {DEFAULT_PROFILE['age']}]: ").strip()
        profile["age"] = int(age_input) if age_input else DEFAULT_PROFILE["age"]

        height_input = input(f"身高 cm [默认: {DEFAULT_PROFILE['height_cm']}]: ").strip()
        profile["height_cm"] = int(height_input) if height_input else DEFAULT_PROFILE["height_cm"]

        weight_input = input(f"体重 kg [默认: {DEFAULT_PROFILE['weight_kg']}]: ").strip()
        profile["weight_kg"] = int(weight_input) if weight_input else DEFAULT_PROFILE["weight_kg"]

        print("\n活动水平选项:")
        print("  1. 久坐办公 (sedentary)")
        print("  2. 轻度活动 (light)")
        print("  3. 中度活动 (moderate)")
        print("  4. 经常运动 (active)")
        print("  5. 高强度运动 (very_active)")
        activity_input = input(f"选择活动水平 [1-5, 默认: 3]: ").strip()
        activity_map = {"1": "sedentary", "2": "light", "3": "moderate", "4": "active", "5": "very_active"}
        profile["activity_level"] = activity_map.get(activity_input, "moderate")

        print("\n目标选项:")
        print("  1. 减脂 (fat_loss)")
        print("  2. 增肌 (muscle_gain)")
        print("  3. 维持体重 (maintain)")
        goal_input = input(f"选择目标 [1-3, 默认: 3]: ").strip()
        goal_map = {"1": "fat_loss", "2": "muscle_gain", "3": "maintain"}
        profile["goal"] = goal_map.get(goal_input, "maintain")

        allergies_input = input("过敏原 (用逗号分隔，没有则回车): ").strip()
        profile["allergies"] = [a.strip() for a in allergies_input.split(",") if a.strip()]

        # 保存
        if save_user_profile(profile):
            daily_goal = get_daily_calorie_goal(profile)
            print("\n✅ 档案已保存！")
            print(f"📊 你的每日热量建议: {daily_goal} 千卡")
        else:
            print("\n⚠️ 保存失败，将使用默认配置")
            profile = DEFAULT_PROFILE.copy()
    except (KeyboardInterrupt, EOFError):
        print("\n\n使用默认配置")
        profile = DEFAULT_PROFILE.copy()
    except Exception as e:
        print(f"\n⚠️ 输入有误: {e}，使用默认配置")
        profile = DEFAULT_PROFILE.copy()

    return profile


# ==============================================
# 主程序
# ==============================================
print("="*80)
print("🥗 交互式营养师助手")
print("="*80)

# 加载或创建用户档案
user_profile = load_user_profile()
profile_exists = os.path.exists(USER_PROFILE_FILE)

if not profile_exists:
    print("\n🔔 检测到首次运行，建议设置个人信息以获得更准确的建议")
    setup_input = input("是否现在设置？(y/n) [默认: y]: ").strip().lower()
    if setup_input != 'n':
        user_profile = setup_user_profile()
        print()
    else:
        save_user_profile(user_profile)
else:
    print(f"\n✅ 已加载用户档案")

print("\n💡 使用说明：")
print("  - 直接输入文字进行对话")
print("  - 提到图片时，写上文件路径，例如：'分析 my_food.jpg'")
print("  - 输入 'profile' 或 '档案' 查看当前个人信息")
print("  - 输入 'setup' 重新设置个人信息")
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

    # 处理特殊命令
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 再见！")
        break

    # 查看档案命令
    if user_input.lower() in ['profile', '档案', '我的档案', '个人信息']:
        profile_text = get_profile_text(user_profile)
        print(f"\n📋 {profile_text}")
        print("-"*80)
        continue

    # 重新设置档案命令
    if user_input.lower() in ['setup', '设置', '重新设置']:
        user_profile = setup_user_profile()
        print("\n" + "-"*80)
        continue

    # 构建消息（附加用户档案）
    profile_text = get_profile_text(user_profile)
    full_message = f"{user_input}\n\n{profile_text}"
    current_state["messages"] = [HumanMessage(content=full_message)]
    
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
