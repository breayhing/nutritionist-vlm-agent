"""
营养师助手 - Streamlit 完整版（多用户支持）
运行：streamlit run app_new.py
"""

import os
import tempfile
import json
import re
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from nutritionist_agent import build_nutritionist_agent
import streamlit as st
from PIL import Image

load_dotenv()

# ==============================================
# 后台处理函数
# ==============================================
def process_agent_in_background(question, agent_state, config, result_container):
    """在后台线程中处理 Agent 推理"""
    try:
        agent = get_agent()
        agent_state["messages"] = [HumanMessage(content=question)]
        result = agent.invoke(agent_state, config=config)
        bot_response = get_final_ai_response(result["messages"])
        result_container["response"] = bot_response
        result_container["agent_state"] = result
    except Exception as e:
        result_container["error"] = str(e)
    finally:
        result_container["complete"] = True
        result_container["processing"] = False

# ==============================================
# 初始化页面级状态
# ==============================================
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = 0

def get_current_user_id():
    """获取当前用户ID（用于防止组件缓存混淆）"""
    return st.session_state.current_user_id

def increment_user_id():
    """增加用户ID，强制刷新所有组件"""
    st.session_state.current_user_id += 1

# ==============================================
# 1. Agent 初始化（缓存，只加载一次）
# ==============================================
@st.cache_resource
def get_agent():
    return build_nutritionist_agent()

def get_final_ai_response(result_messages):
    """提取最终 AI 回复（跳过工具调用消息）"""
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "抱歉，我暂时无法回复。"

# ==============================================
# 2. 多用户管理
# ==============================================
USERS_DIR = "users"

def get_current_username():
    """获取当前用户名"""
    return st.session_state.get("current_username", "default_user")

def get_safe_username(username):
    """将用户名转换为安全的文件名"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))

def get_user_file_path(filename, username=None):
    """获取指定用户的文件路径"""
    if username is None:
        username = get_current_username()
    safe_username = get_safe_username(username)
    return os.path.join(USERS_DIR, f"{filename}_{safe_username}.json")

def get_all_users():
    """获取所有已注册的用户列表"""
    users = set()
    if not os.path.exists(USERS_DIR):
        return []
    try:
        for filename in os.listdir(USERS_DIR):
            if filename.startswith("user_profile_") and filename.endswith(".json"):
                # 提取用户名
                username = filename.replace("user_profile_", "").replace(".json", "")
                users.add(username)
    except:
        pass
    return sorted(list(users))

# ==============================================
# 3. 用户档案管理（多用户隔离）
# ==============================================

def init_user_profile():
    """初始化当前用户的档案 - 只在没有 session 数据时从文件加载"""
    current_user = get_current_username()

    # 如果 session 中已有档案，直接返回（不重新加载）
    if "user_profile" in st.session_state:
        print(f"📋 使用内存中的用户 [{current_user}] 档案")
        return

    # 默认档案
    default_profile = {
        "height_cm": 175,
        "weight_kg": 70,
        "age": 25,
        "gender": "male",
        "activity_level": "moderate",
        "goal": "maintain",
        "allergies": [],
        "manual_bmr": None,
        "manual_tdee": None,
        "manual_daily_goal": None
    }

    # 确保用户目录存在
    os.makedirs(USERS_DIR, exist_ok=True)

    safe_username = get_safe_username(current_user)
    profile_file = os.path.join(USERS_DIR, f"user_profile_{safe_username}.json")

    print(f"🔍 [DEBUG] init_user_profile: current_user={current_user}, profile_file={profile_file}, exists={os.path.exists(profile_file)}")

    # 从文件加载
    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                loaded_profile = json.load(f)
                print(f"📖 [DEBUG] 文件内容: {loaded_profile}")
                # 兼容旧档案，添加新字段
                for key in ["manual_bmr", "manual_tdee", "manual_daily_goal"]:
                    if key not in loaded_profile:
                        loaded_profile[key] = None
                # 移除旧字段
                if "use_manual_values" in loaded_profile:
                    del loaded_profile["use_manual_values"]
                st.session_state.user_profile = loaded_profile
            print(f"✅ 已从文件加载用户 [{current_user}] 的档案: {loaded_profile.get('height_cm')}cm, {loaded_profile.get('weight_kg')}kg")
            return
        except Exception as e:
            print(f"⚠️ 加载用户档案失败，使用默认值: {e}")
            import traceback
            traceback.print_exc()

    # 文件不存在或加载失败，使用默认值
    print(f"📝 创建新用户 [{current_user}] 的默认档案")
    st.session_state.user_profile = default_profile
    save_user_profile()

def save_user_profile():
    """保存当前用户档案到文件"""
    try:
        current_user = get_current_username()
        os.makedirs(USERS_DIR, exist_ok=True)
        profile_file = get_user_file_path("user_profile", username=current_user)
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.user_profile, f, ensure_ascii=False, indent=2)
        print(f"✅ 用户 [{current_user}] 档案已保存到: {profile_file}")
        print(f"   内容: {st.session_state.user_profile.get('height_cm')}cm, {st.session_state.user_profile.get('weight_kg')}kg")
        return True
    except Exception as e:
        print(f"❌ 保存用户档案失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_user_profile(username):
    """加载指定用户的档案"""
    safe_username = get_safe_username(username)
    profile_file = os.path.join(USERS_DIR, f"user_profile_{safe_username}.json")

    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass

    # 返回默认档案
    return {
        "height_cm": 175,
        "weight_kg": 70,
        "age": 25,
        "gender": "male",
        "activity_level": "moderate",
        "goal": "maintain",
        "allergies": []
    }

def get_user_profile_text():
    """获取用户档案的文本描述，用于传递给 Agent"""
    profile = st.session_state.user_profile
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

    username = get_current_username()

    # 获取今日已摄入统计
    today_summary = get_today_summary()
    remaining_calories = daily_goal - today_summary['total_calories']

    profile_text = f"""【用户档案】
- 用户名: {username}
- 基本信息: {gender_text}, {profile['age']}岁, 身高{profile['height_cm']}cm, 体重{profile['weight_kg']}kg
- 活动水平: {activity_text}
- 健康目标: {goal_text}
- 每日建议热量: {daily_goal} 千卡
- 过敏原: {', '.join(profile['allergies']) if profile['allergies'] else '无'}

【今日已摄入】
- 已摄入热量: {today_summary['total_calories']} 千卡
- 剩余热量: {max(0, remaining_calories)} 千卡
- 蛋白质: {today_summary['total_protein']:.0f}g
- 碳水: {today_summary['total_carbs']:.0f}g
- 脂肪: {today_summary['total_fat']:.0f}g
- 餐数: {today_summary['meal_count']}餐

【重要】记住这个用户档案！在所有回复中都要考虑用户的个人信息、目标和今日已摄入情况。规划饮食时必须考虑剩余热量。"""

    return profile_text

# ==============================================
# 4. 对话历史管理（多用户隔离）
# ==============================================

def init_chat_history():
    """初始化当前用户的对话历史"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_history_file = get_user_file_path("chat_history")

    # 只在首次加载时从文件读取
    if "chat_history_loaded" not in st.session_state:
        if os.path.exists(chat_history_file):
            try:
                with open(chat_history_file, "r", encoding="utf-8") as f:
                    st.session_state.messages = json.load(f)
                print(f"✅ 已加载用户 [{get_current_username()}] 的对话历史")
            except Exception as e:
                print(f"⚠️ 加载对话历史失败: {e}")
        st.session_state.chat_history_loaded = True

def save_chat_history():
    """保存当前用户的对话历史"""
    try:
        os.makedirs(USERS_DIR, exist_ok=True)
        chat_history_file = get_user_file_path("chat_history")
        with open(chat_history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        print(f"✅ 用户 [{get_current_username()}] 对话历史已保存")
        return True
    except Exception as e:
        print(f"❌ 保存对话历史失败: {e}")
        return False

def load_chat_history(username):
    """加载指定用户的对话历史"""
    safe_username = get_safe_username(username)
    chat_history_file = os.path.join(USERS_DIR, f"chat_history_{safe_username}.json")

    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []

# ==============================================
# 5. 饮食历史管理（多用户隔离）
# ==============================================

def init_meal_history():
    """初始化当前用户的饮食历史"""
    if "meal_history" not in st.session_state:
        st.session_state.meal_history = []

    meal_history_file = get_user_file_path("meal_history")

    # 只在首次加载时从文件读取
    if "meal_history_loaded" not in st.session_state:
        if os.path.exists(meal_history_file):
            try:
                with open(meal_history_file, "r", encoding="utf-8") as f:
                    st.session_state.meal_history = json.load(f)
                print(f"✅ 已加载用户 [{get_current_username()}] 的饮食历史")
            except Exception as e:
                print(f"⚠️ 加载饮食历史失败: {e}")
        st.session_state.meal_history_loaded = True

def save_meal_record(meal_data):
    """保存饮食记录"""
    # 如果有图片，复制一份到用户图片目录
    image_path = meal_data.get("image_path", "")
    saved_image_path = ""

    if image_path and os.path.exists(image_path):
        try:
            # 读取原始图片
            with open(image_path, "rb") as f:
                img_data = f.read()

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_image_path = os.path.join(get_user_images_dir(), f"meal_{timestamp}.jpg")

            # 保存到用户目录
            with open(saved_image_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            print(f"⚠️ 保存图片失败: {e}")

    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "foods": meal_data.get("foods", []),
        "total_calories": meal_data.get("total_calories", 0),
        "protein_g": meal_data.get("protein_g", 0),
        "carbs_g": meal_data.get("carbs_g", 0),
        "fat_g": meal_data.get("fat_g", 0),
        "image_path": saved_image_path  # 保存到用户目录的路径
    }
    st.session_state.meal_history.append(record)

    # 保存到文件
    try:
        os.makedirs(USERS_DIR, exist_ok=True)
        meal_history_file = get_user_file_path("meal_history")
        with open(meal_history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.meal_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 保存饮食历史失败: {e}")

def get_today_summary():
    """获取今日饮食总结"""
    today = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in st.session_state.meal_history if r["date"].startswith(today)]

    return {
        "total_calories": sum(r["total_calories"] for r in today_records),
        "total_protein": sum(r["protein_g"] for r in today_records),
        "total_carbs": sum(r["carbs_g"] for r in today_records),
        "total_fat": sum(r["fat_g"] for r in today_records),
        "meal_count": len(today_records)
    }

# ==============================================
# 6. 用户切换功能
# ==============================================

def switch_user(new_username):
    """切换到指定用户"""
    if not new_username or new_username.strip() == "":
        return False

    new_username = new_username.strip()

    # 如果用户名没变，不做处理
    if new_username == get_current_username():
        return False

    # 1. 保存当前用户的所有数据
    old_username = get_current_username()
    print(f"🔄 切换用户: [{old_username}] -> [{new_username}]")

    if "user_profile" in st.session_state:
        print(f"💾 保存 [{old_username}] 档案")
        save_user_profile()

    if "messages" in st.session_state:
        save_chat_history()

    # 2. 增加用户ID，强制刷新所有组件
    increment_user_id()

    # 3. 切换用户名
    st.session_state.current_username = new_username

    # 4. 清除所有用户相关的 session 状态
    keys_to_delete = [
        "user_profile", "messages", "meal_history",
        "chat_history_loaded", "meal_history_loaded",
        "profile_modified", "edit_bmr", "edit_tdee", "edit_goal",
        "last_uploaded_file", "uploader_key"
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    # 5. 重置 agent 状态
    st.session_state.agent_state = {
        "messages": [],
        "image_analyzed": False,
        "image_info": {}
    }

    print(f"✅ 已切换到用户 [{new_username}]，用户ID={get_current_user_id()}")
    return True

# ==============================================
# 7. 辅助计算函数
# ==============================================

def calculate_bmr(profile):
    weight = profile["weight_kg"]
    height = profile["height_cm"]
    age = profile["age"]
    if profile["gender"] == "male":
        return int(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age))
    else:
        return int(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))

def calculate_tdee(bmr, activity_level):
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    return int(bmr * multipliers.get(activity_level, 1.55))

def get_daily_calorie_goal(profile):
    # 如果用户手动设置了每日目标，优先使用
    manual_goal = profile.get("manual_daily_goal")
    if manual_goal and manual_goal > 0:
        return int(manual_goal)

    # 否则计算
    bmr = get_actual_bmr(profile)
    tdee = get_actual_tdee(profile)

    if profile["goal"] == "fat_loss":
        return int(tdee * 0.8)
    elif profile["goal"] == "muscle_gain":
        return int(tdee * 1.1)
    else:
        return tdee

def get_actual_bmr(profile):
    """获取实际 BMR（优先使用手动设置的值）"""
    manual_bmr = profile.get("manual_bmr")
    if manual_bmr and manual_bmr > 0:
        return int(manual_bmr)
    return calculate_bmr(profile)

def get_actual_tdee(profile):
    """获取实际 TDEE（优先使用手动设置的值）"""
    manual_tdee = profile.get("manual_tdee")
    if manual_tdee and manual_tdee > 0:
        return int(manual_tdee)

    bmr = get_actual_bmr(profile)
    return calculate_tdee(bmr, profile["activity_level"])

def get_bmr_reference(age, gender):
    """获取同龄人的平均 BMR 参考值"""
    # 使用 Mifflin-St Jeor 公式的简化平均值计算
    # 男性平均：175cm, 70kg; 女性平均：165cm, 55kg
    if gender == "male":
        base_weight = 70
        base_height = 175
        base_bmr = int(88.362 + (13.397 * base_weight) + (4.799 * base_height) - (5.677 * age))
    else:
        base_weight = 55
        base_height = 165
        base_bmr = int(447.593 + (9.247 * base_weight) + (3.098 * base_height) - (4.330 * age))
    return base_bmr

def get_tdee_reference(age, gender):
    """获取同龄人的平均 TDEE 参考值（假设中度活动）"""
    bmr = get_bmr_reference(age, gender)
    return int(bmr * 1.55)  # 中度活动

def clear_today_meals():
    """清空今日的饮食记录"""
    today = datetime.now().strftime("%Y-%m-%d")
    # 保留非今日的记录
    filtered_history = [r for r in st.session_state.meal_history if not r["date"].startswith(today)]
    st.session_state.meal_history = filtered_history

    # 保存到文件
    try:
        os.makedirs(USERS_DIR, exist_ok=True)
        meal_history_file = get_user_file_path("meal_history")
        with open(meal_history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.meal_history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 保存饮食历史失败: {e}")
        return False

# ==============================================
# 8. 用户图片管理
# ==============================================
def get_user_images_dir():
    """获取当前用户的图片目录"""
    safe_username = get_safe_username(get_current_username())
    user_images_dir = os.path.join(USERS_DIR, f"images_{safe_username}")
    os.makedirs(user_images_dir, exist_ok=True)
    return user_images_dir

def save_uploaded_image(uploaded_file):
    """保存用户上传的图片，返回保存路径"""
    user_images_dir = get_user_images_dir()

    # 生成唯一文件名：时间戳 + 原文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = uploaded_file.name
    file_ext = os.path.splitext(original_name)[1] or ".jpg"
    safe_filename = f"{timestamp}_{re.sub(r'[^\w.]', '_', original_name)}"

    file_path = os.path.join(user_images_dir, safe_filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def get_user_images():
    """获取当前用户保存的图片列表"""
    user_images_dir = get_user_images_dir()
    if not os.path.exists(user_images_dir):
        return []

    images = []
    for filename in sorted(os.listdir(user_images_dir), reverse=True):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            file_path = os.path.join(user_images_dir, filename)
            # 获取文件信息
            stat = os.stat(file_path)
            images.append({
                "name": filename,
                "path": file_path,
                "size": stat.st_size,
                "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            })
    return images

def delete_user_image(filename):
    """删除用户指定的图片"""
    user_images_dir = get_user_images_dir()
    file_path = os.path.join(user_images_dir, filename)

    print(f"🗑️ [DEBUG] 尝试删除图片: filename={filename}, file_path={file_path}, exists={os.path.exists(file_path)}")

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✅ 图片已删除: {filename}")
            return True
        except Exception as e:
            print(f"❌ 删除图片失败: {e}")
            return False
    else:
        print(f"⚠️ 文件不存在: {file_path}")
    return False

def delete_and_rerun(filename):
    """删除图片并重新运行页面"""
    if delete_user_image(filename):
        st.toast("✅ 图片已删除")
    st.rerun()

# ==============================================
# 9. 页面配置
# ==============================================
st.set_page_config(
    page_title="🥗 营养师助手",
    page_icon="🥗",
    layout="wide"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1rem;
    }
    .user-switcher {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">🥗 营养师助手</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">上传食物图片，AI 帮你识别食材、计算热量、给出营养建议</p>', unsafe_allow_html=True)

# ==============================================
# 9. 用户切换区域（页面顶部）
# ==============================================

# 初始化当前用户名
if "current_username" not in st.session_state:
    st.session_state.current_username = "default_user"

st.markdown('<div class="user-switcher">', unsafe_allow_html=True)

col_user1, col_user2, col_user3 = st.columns([2, 2, 4])

with col_user1:
    current_user = st.text_input(
        "👤 当前用户",
        value=get_current_username(),
        key="username_input",
        placeholder="输入用户名"
    )

with col_user2:
    st.write("")  # 对齐
    st.write("")
    if st.button("🔄 切换用户", use_container_width=True):
        if switch_user(current_user):
            st.rerun()

with col_user3:
    st.write("")  # 对齐
    st.write("")
    all_users = get_all_users()
    if all_users:
        st.caption(f"已注册用户: {', '.join(all_users[:5])}{'...' if len(all_users) > 5 else ''}")
    else:
        st.caption("暂无其他用户")

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================
# 10. 初始化会话状态
# ==============================================
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "image_analyzed": False,
        "image_info": {}
    }

# 推理相关状态
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "agent_response" not in st.session_state:
    st.session_state.agent_response = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = True
if "thinking_start_time" not in st.session_state:
    st.session_state.thinking_start_time = None

# 初始化用户数据
init_user_profile()
init_chat_history()
init_meal_history()

# 检查档案是否被外部更新（仅在调试时启用）
current_user = get_current_username()
profile_file = get_user_file_path("user_profile", username=current_user)
if os.path.exists(profile_file):
    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            file_profile = json.load(f)
        if file_profile != st.session_state.user_profile:
            print(f"🔄 检测到档案被外部更新，重新加载")
            st.session_state.user_profile = file_profile
    except Exception as e:
        print(f"⚠️ 检查外部更新失败: {e}")

# ==============================================
# 11. 侧边栏
# ==============================================
# 11. 侧边栏
# ==============================================
with st.sidebar:
    st.markdown("### 📸 上传食物图片")

    # 使用 key 来确保上传状态正确管理
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "点击或拖拽上传图片",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    image_path = None
    if uploaded_file is not None:
        # 获取上传文件的唯一标识（文件名 + 大小）
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        last_uploaded = st.session_state.get("last_uploaded_file")

        # 只有当文件是新上传的时候才保存
        if file_id != last_uploaded:
            saved_path = save_uploaded_image(uploaded_file)
            st.session_state.last_uploaded_file = file_id
            st.toast("✅ 图片已保存到你的图库")

        # 同时保存到临时目录供 Agent 使用
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "food_image.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="已上传图片", use_container_width=True)
    else:
        st.info("⚡ 等待上传图片")

    st.markdown("---")

    # ----- 图片历史管理（折叠区）-----
    with st.expander("🖼️ 我的图库", expanded=False):
        user_images = get_user_images()

        if user_images:
            st.caption(f"共 {len(user_images)} 张图片")

            # 显示最近的图片（最多6张）
            for img in user_images[:6]:
                col_img1, col_img2 = st.columns([3, 1])
                with col_img1:
                    st.image(img["path"], use_container_width=True)
                    st.caption(f"📅 {img['time']}")
                with col_img2:
                    # 使用 on_click 回调来确保删除逻辑只执行一次
                    st.button("🗑️", key=f"del_{img['name']}", help="删除这张图片",
                              on_click=lambda f=img["name"]: delete_and_rerun(f), use_container_width=True)

            if len(user_images) > 6:
                st.caption(f"...还有 {len(user_images) - 6} 张图片")

            # 清空所有图片按钮（两步确认）
            if "confirm_clear_images" not in st.session_state:
                st.session_state.confirm_clear_images = False

            if st.session_state.confirm_clear_images:
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✓ 确认清空", use_container_width=True, type="primary"):
                        user_images_dir = get_user_images_dir()
                        import shutil
                        try:
                            shutil.rmtree(user_images_dir)
                            os.makedirs(user_images_dir)
                            st.session_state.confirm_clear_images = False
                            st.toast("✅ 所有图片已清空")
                            time.sleep(0.3)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 清空失败: {e}")
                with col_no:
                    if st.button("✗ 取消", use_container_width=True):
                        st.session_state.confirm_clear_images = False
                        st.rerun()
            else:
                if st.button("🗑️ 清空所有图片", use_container_width=True, type="secondary"):
                    st.session_state.confirm_clear_images = True
                    st.rerun()
        else:
            st.info("暂无保存的图片")
            st.caption("上传的图片会自动保存在这里")

    st.markdown("---")

    # ----- 用户档案（折叠区）-----
    with st.expander("👤 我的档案", expanded=False):
        if "profile_modified" not in st.session_state:
            st.session_state.profile_modified = False

        st.caption(f"👤 用户: **{get_current_username()}**")

        # 获取当前用户ID，用于组件key
        user_id = get_current_user_id()

        col1, col2 = st.columns(2)
        with col1:
            new_height = st.number_input(
                "身高(cm)", min_value=100, max_value=250,
                value=st.session_state.user_profile["height_cm"],
                key=f"height_{user_id}"
            )
            if new_height != st.session_state.user_profile["height_cm"]:
                st.session_state.user_profile["height_cm"] = new_height
                st.session_state.profile_modified = True

            new_age = st.number_input(
                "年龄", min_value=15, max_value=100,
                value=st.session_state.user_profile["age"],
                key=f"age_{user_id}"
            )
            if new_age != st.session_state.user_profile["age"]:
                st.session_state.user_profile["age"] = new_age
                st.session_state.profile_modified = True

        with col2:
            new_weight = st.number_input(
                "体重(kg)", min_value=30, max_value=200,
                value=st.session_state.user_profile["weight_kg"],
                key=f"weight_{user_id}"
            )
            if new_weight != st.session_state.user_profile["weight_kg"]:
                st.session_state.user_profile["weight_kg"] = new_weight
                st.session_state.profile_modified = True

            new_gender = st.selectbox(
                "性别", ["male", "female"],
                index=0 if st.session_state.user_profile["gender"] == "male" else 1,
                key=f"gender_{user_id}"
            )
            if new_gender != st.session_state.user_profile["gender"]:
                st.session_state.user_profile["gender"] = new_gender
                st.session_state.profile_modified = True

        new_activity = st.selectbox(
            "活动水平",
            ["sedentary", "light", "moderate", "active", "very_active"],
            format_func=lambda x: {
                "sedentary": "久坐办公", "light": "轻度活动",
                "moderate": "中度活动", "active": "经常运动",
                "very_active": "高强度运动"
            }[x],
            index=["sedentary", "light", "moderate", "active", "very_active"].index(
                st.session_state.user_profile["activity_level"]
            ),
            key=f"activity_{user_id}"
        )
        if new_activity != st.session_state.user_profile["activity_level"]:
            st.session_state.user_profile["activity_level"] = new_activity
            st.session_state.profile_modified = True

        new_goal = st.selectbox(
            "我的目标",
            ["fat_loss", "muscle_gain", "maintain"],
            format_func=lambda x: {"fat_loss": "🔥 减脂", "muscle_gain": "💪 增肌", "maintain": "⚖️ 维持"}[x],
            index=["fat_loss", "muscle_gain", "maintain"].index(
                st.session_state.user_profile["goal"]
            ),
            key=f"goal_{user_id}"
        )
        if new_goal != st.session_state.user_profile["goal"]:
            st.session_state.user_profile["goal"] = new_goal
            st.session_state.profile_modified = True

        allergies_input = st.text_input(
            "过敏原（用逗号分隔）",
            value=", ".join(st.session_state.user_profile["allergies"]),
            key=f"allergies_{user_id}"
        )
        new_allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
        if new_allergies != st.session_state.user_profile["allergies"]:
            st.session_state.user_profile["allergies"] = new_allergies
            st.session_state.profile_modified = True

        # 显示计算结果（带中文解释和参考值）
        bmr = get_actual_bmr(st.session_state.user_profile)
        tdee = get_actual_tdee(st.session_state.user_profile)
        daily_goal = get_daily_calorie_goal(st.session_state.user_profile)

        # 获取参考值
        ref_bmr = get_bmr_reference(
            st.session_state.user_profile["age"],
            st.session_state.user_profile["gender"]
        )
        ref_tdee = get_tdee_reference(
            st.session_state.user_profile["age"],
            st.session_state.user_profile["gender"]
        )

        st.markdown("---")
        st.markdown("### 📊 代谢数据")

        # 每个指标独立显示，带修改按钮
        col_m1, col_m2, col_m3 = st.columns(3)

        # BMR 指标
        with col_m1:
            st.metric("BMR", f"{bmr}", help="基础代谢率：身体静止时的最低热量消耗")

            # 修改按钮
            if st.button("✏️ 修改", key=f"btn_bmr_{st.session_state.get('rerun_count', 0)}", use_container_width=True):
                st.session_state.edit_bmr = True

            # 展开/收起输入框
            if st.session_state.get("edit_bmr", False):
                manual_bmr = st.session_state.user_profile.get("manual_bmr")
                current_value = manual_bmr if manual_bmr and manual_bmr > 0 else bmr
                new_bmr = st.number_input(
                    "输入 BMR 值",
                    min_value=1000,
                    max_value=4000,
                    value=current_value,
                    key="input_bmr",
                    help="输入新值或保持不变"
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("保存", key="save_bmr", use_container_width=True):
                        st.session_state.user_profile["manual_bmr"] = int(new_bmr)
                        st.session_state.profile_modified = True
                        st.session_state.edit_bmr = False
                        save_user_profile()
                        st.rerun()
                with col_cancel:
                    if st.button("恢复默认", key="reset_bmr", use_container_width=True):
                        st.session_state.user_profile["manual_bmr"] = None
                        st.session_state.profile_modified = True
                        st.session_state.edit_bmr = False
                        save_user_profile()
                        st.rerun()

        # TDEE 指标
        with col_m2:
            st.metric("TDEE", f"{tdee}", help="每日总消耗：包含活动和基础代谢")

            if st.button("✏️ 修改", key=f"btn_tdee_{st.session_state.get('rerun_count', 0)}", use_container_width=True):
                st.session_state.edit_tdee = True

            if st.session_state.get("edit_tdee", False):
                manual_tdee = st.session_state.user_profile.get("manual_tdee")
                current_value = manual_tdee if manual_tdee and manual_tdee > 0 else tdee
                new_tdee = st.number_input(
                    "输入 TDEE 值",
                    min_value=1200,
                    max_value=5000,
                    value=current_value,
                    key="input_tdee"
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("保存", key="save_tdee", use_container_width=True):
                        st.session_state.user_profile["manual_tdee"] = int(new_tdee)
                        st.session_state.profile_modified = True
                        st.session_state.edit_tdee = False
                        save_user_profile()
                        st.rerun()
                with col_cancel:
                    if st.button("恢复默认", key="reset_tdee", use_container_width=True):
                        st.session_state.user_profile["manual_tdee"] = None
                        st.session_state.profile_modified = True
                        st.session_state.edit_tdee = False
                        save_user_profile()
                        st.rerun()

        # 摄入热量指标
        with col_m3:
            calculated_goal = daily_goal
            manual_goal = st.session_state.user_profile.get("manual_daily_goal")

            if manual_goal and manual_goal > 0:
                display_goal = manual_goal
                st.metric("摄入热量", f"{display_goal}", help="每日建议摄入的最高热量")
            else:
                st.metric("摄入热量", f"{calculated_goal}", help="每日建议摄入的最高热量")

            if st.button("✏️ 修改", key=f"btn_goal_{st.session_state.get('rerun_count', 0)}", use_container_width=True):
                st.session_state.edit_goal = True

            if st.session_state.get("edit_goal", False):
                current_value = manual_goal if manual_goal and manual_goal > 0 else calculated_goal
                new_goal = st.number_input(
                    "输入每日摄入热量",
                    min_value=500,
                    max_value=5000,
                    value=current_value,
                    key="input_goal"
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("保存", key="save_goal", use_container_width=True):
                        st.session_state.user_profile["manual_daily_goal"] = int(new_goal)
                        st.session_state.profile_modified = True
                        st.session_state.edit_goal = False
                        save_user_profile()
                        st.rerun()
                with col_cancel:
                    if st.button("恢复默认", key="reset_goal", use_container_width=True):
                        st.session_state.user_profile["manual_daily_goal"] = None
                        st.session_state.profile_modified = True
                        st.session_state.edit_goal = False
                        save_user_profile()
                        st.rerun()

        # 提示信息
        has_manual_values = any([
            st.session_state.user_profile.get("manual_bmr"),
            st.session_state.user_profile.get("manual_tdee"),
            st.session_state.user_profile.get("manual_daily_goal")
        ])
        if has_manual_values:
            st.info("💡 已设置手动值：数据来源于运动手表、体脂秤等设备。点击「修改」可调整或恢复默认值。")

        # 保存按钮
        if st.session_state.profile_modified:
            if st.button("💾 保存档案", use_container_width=True, type="primary"):
                if save_user_profile():
                    st.session_state.profile_modified = False
                    st.success("✅ 档案已保存！")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 保存失败，请重试")
        else:
            st.info("✅ 档案已保存")

        # 档案最后更新时间
        profile_file = get_user_file_path("user_profile")
        if os.path.exists(profile_file):
            file_mtime = os.path.getmtime(profile_file)
            last_update = time.strftime("%Y-%m-%d %H:%M", time.localtime(file_mtime))
            st.caption(f"最后更新: {last_update}")

    # ----- 今日统计（折叠区）-----
    with st.expander("📊 今日统计", expanded=False):
        today_summary = get_today_summary()
        daily_goal = get_daily_calorie_goal(st.session_state.user_profile)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("已摄入", f"{today_summary['total_calories']}/{daily_goal}")
            remaining = daily_goal - today_summary['total_calories']
            st.metric("剩余", f"{max(0, remaining)}" if remaining > 0 else "已超量")
        with col2:
            st.metric("餐数", f"{today_summary['meal_count']}")

        st.markdown("**营养素**")
        st.markdown(f"蛋白质: {today_summary['total_protein']:.0f}g")
        st.markdown(f"碳水: {today_summary['total_carbs']:.0f}g")
        st.markdown(f"脂肪: {today_summary['total_fat']:.0f}g")

        st.markdown("---")
        if st.button("🗑️ 清空今日记录", use_container_width=True, type="secondary"):
            if clear_today_meals():
                st.success("✅ 今日记录已清空")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ 清空失败")

    st.markdown("---")
    st.markdown("### 💡 试试这些问题")
    st.markdown("""
    - 分析这张图片
    - 这餐热量高吗？
    - 适合减脂期吃吗？
    - 蛋白质够不够？
    - 有什么建议吗？
    """)

    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        save_chat_history()
        st.session_state.agent_state = {
            "messages": [],
            "image_analyzed": False,
            "image_info": {}
        }
        st.rerun()

# ==============================================
# 12. 主界面：显示历史消息
# ==============================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🥗"):
            st.markdown(msg["content"])

# ==============================================
# 13. 显示 AI 思考状态
# ==============================================
# 创建占位符用于显示思考状态和结果
thinking_placeholder = st.empty()

if st.session_state.pending_question and not st.session_state.processing_complete:
    # 检查后台任务是否完成
    if st.session_state.agent_result.get("complete", False):
        st.session_state.processing_complete = True
        # 任务完成，立即刷新显示结果
        st.rerun()
    else:
        with thinking_placeholder.container():
            st.info("🤔 AI 思考中...")
        # 任务进行中，使用轻量级自动刷新
        time.sleep(1)
        st.rerun()

# ==============================================
# 14. 处理后台推理结果
# ==============================================
if st.session_state.pending_question and st.session_state.processing_complete:
    # 获取后台处理的结果
    result = st.session_state.agent_result
    prompt = st.session_state.pending_question

    # 清除思考状态
    thinking_placeholder.empty()

    # 清除状态
    st.session_state.pending_question = None
    st.session_state.processing_complete = False
    st.session_state.thinking_start_time = None

    # 显示 AI 回复
    with st.chat_message("assistant", avatar="🥗"):
        if "error" in result:
            st.error(f"❌ 出错啦：{result['error']}")
            st.session_state.messages.append({"role": "assistant", "content": f"❌ 出错啦：{result['error']}"})
        else:
            bot_response = result["response"]
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.agent_state.update(result["agent_state"])

            # 保存对话历史
            save_chat_history()

            # 重新加载 meal_history（AI可能调用了record_meal工具）
            meal_history_file = get_user_file_path("meal_history")
            if os.path.exists(meal_history_file):
                try:
                    with open(meal_history_file, "r", encoding="utf-8") as f:
                        st.session_state.meal_history = json.load(f)
                    print(f"✅ 已重新加载饮食历史")
                except Exception as e:
                    print(f"⚠️ 重新加载饮食历史失败: {e}")

            # 刷新页面以更新侧边栏统计
            time.sleep(0.3)
            st.rerun()

    # 清理 agent_result
    st.session_state.agent_result = {}

# ==============================================
# 15. 处理用户输入
# ==============================================
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 构建完整消息
    profile_text = get_user_profile_text()

    if image_path:
        full_message = f"{prompt}\n\n{profile_text}\n[图片路径: {image_path}]"
    else:
        full_message = f"{prompt}\n\n{profile_text}"

    print(f"📝 发送给 Agent 的消息: {full_message[:200]}...")

    # 设置后台处理状态
    st.session_state.pending_question = prompt
    st.session_state.processing_complete = False
    st.session_state.thinking_start_time = time.time()

    # 准备后台处理的结果容器
    st.session_state.agent_result = {
        "complete": False,
        "processing": True
    }

    # 启动后台线程处理
    thread = threading.Thread(
        target=process_agent_in_background,
        args=(full_message, st.session_state.agent_state.copy(),
              {"configurable": {"thread_id": f"streamlit_chat_{get_current_username()}"}},
              st.session_state.agent_result)
    )
    thread.daemon = True
    thread.start()

    # 立即刷新以显示思考状态
    st.rerun()

