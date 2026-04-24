"""
================================================================================
【正式主程序】营养师助手 - 完整功能实现
================================================================================

用途：这是项目的核心文件，包含完整的 Agent 架构和功能实现

主要功能：
1. 双模型架构（Qwen-VL-Max 视觉 + Qwen-Max 推理）
2. 三大工具（视觉分析、热量计算、营养建议）
3. LangGraph 工作流（支持多轮对话记忆）
4. 自定义分析指令（Agent 自主决定如何分析图片）

如何使用：
- 方式1（推荐）：运行 chat_demo.py 进行交互式对话
- 方式2：直接运行本文件查看演示（python nutritionist_agent.py）
- 方式3：在其他程序中导入 build_nutritionist_agent() 函数

依赖：
- .env 文件中需要配置 DASHSCOPE_API_KEY

注意：
- 本文件底部的 if __name__ == "__main__" 是演示代码
- 实际使用时建议通过 chat_demo.py 进行交互
================================================================================
"""

import os
import json
from typing import Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import dashscope
import base64
from PIL import Image

# 加载环境变量
load_dotenv()

# ==============================================
# 【配置】API Keys
# ==============================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ==============================================
# 【工具定义】标准化的 LangChain Tools
# ==============================================

@tool
def analyze_food_image(image_path: str, analysis_instruction: str = "请分析这张图片中的食物，并以 JSON 格式返回以下信息：\n\n{\n    \"foods\": [\n        {\n            \"name\": \"食物名称\",\n            \"estimated_weight_g\": 估算重量(克),\n            \"calories_per_100g\": 每100克热量(千卡),\n            \"estimated_calories\": 估算热量(千卡),\n            \"protein_g\": 蛋白质含量(克),\n            \"carbs_g\": 碳水化合物含量(克),\n            \"fat_g\": 脂肪含量(克)\n        }\n    ],\n    \"total_estimated_calories\": 总热量估算(千卡),\n    \"total_protein_g\": 总蛋白质(克),\n    \"total_carbs_g\": 总碳水(克),\n    \"total_fat_g\": 总脂肪(克),\n    \"nutrition_tags\": [\"高蛋白\", \"低脂\", \"高纤维\"等标签]\n}\n\n要求：\n1. 尽可能准确地识别所有食材\n2. 基于常见份量估算重量\n3. 使用标准的营养数据库值，估算蛋白质、碳水、脂肪含量\n4. 必须返回完整的营养素信息（蛋白质、碳水、脂肪）\n5. 只返回 JSON，不要有其他文字") -> str:
    """
    分析食物图片，识别食材并估算营养信息
    
    Args:
        image_path: 图片的本地文件路径（例如："test.jpg" 或 "C:/images/food.png"）
        analysis_instruction: 分析指令，告诉模型如何分析图片。默认是标准的营养成分分析。
                             你可以根据需要自定义，例如：
                             - "只识别食材名称，不需要热量"
                             - "重点分析蛋白质含量"
                             - "判断这是否适合减脂期食用"
    
    Returns:
        根据分析指令返回相应的结果（通常是 JSON 格式）
    """
    try:
        print(f"\n🔍 [视觉工具] 正在分析图片: {image_path}")
        print(f"📝 [视觉工具] 分析指令: {analysis_instruction[:100]}...")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return f"错误：找不到图片文件 {image_path}"
        
        # 转换为 base64
        image_base64 = image_to_base64(image_path)
        
        # 配置 Qwen-VL
        dashscope.api_key = DASHSCOPE_API_KEY
        
        messages = [
            {
                'role': 'system',
                'content': [{'text': '你是专业的营养师助手，擅长食物识别和营养分析'}]
            },
            {
                'role': 'user',
                'content': [
                    {'image': f"data:image/jpeg;base64,{image_base64}"},
                    {'text': analysis_instruction}
                ]
            }
        ]
        
        # 调用 Qwen-VL
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages
        )
        
        if response.status_code != 200:
            error_msg = f"视觉分析失败: {response.code} - {response.message}"
            print(f"❌ [视觉工具] {error_msg}")
            return error_msg
        
        # 解析结果
        result_text = response.output.choices[0].message.content[0]['text']
        print(f"✅ [视觉工具] 分析完成")
        print(f"📊 [视觉工具] 结果预览: {result_text[:150]}...")
        
        return result_text
        
    except Exception as e:
        error_msg = f"视觉分析错误: {str(e)}"
        print(f"❌ [视觉工具] {error_msg}")
        return error_msg


@tool
def calculate_total_calories(calories_list: str) -> str:
    """
    计算多个食物的总热量
    
    Args:
        calories_list: 热量列表，可以是以下格式之一：
                       - 加法表达式: "100+200+50"
                       - 数字列表: "100,200,50"
                       - 数字字符串: "350"
    
    Returns:
        总热量数值的字符串
    """
    try:
        import re
        
        # 清理输入
        calories_str = str(calories_list).strip()
        
        # 如果输入是 JSON 字符串，尝试提取数字
        if calories_str.startswith('{') or calories_str.startswith('```'):
            # 尝试提取数字
            numbers = re.findall(r'\d+', calories_str)
            if numbers:
                calories_str = '+'.join(numbers)
        
        # 替换中文逗号为英文逗号
        calories_str = calories_str.replace('，', ',')
        
        # 如果是逗号分隔，转成加号
        if ',' in calories_str and '+' not in calories_str:
            calories_str = calories_str.replace(',', '+')
        
        # 提取数字和运算符
        cleaned = re.findall(r'[\d+\-\*\/\.]+', calories_str)
        if cleaned:
            result = eval(cleaned[0])
            return f"总热量：{int(result)} 千卡"
        else:
            return f"无法解析热量值：{calories_list}"
    except Exception as e:
        return f"计算错误：{str(e)}，输入为：{calories_list}"

@tool
def get_nutrition_advice(total_calories: int, food_types: str, username: str = "default_user", user_context: str = "") -> str:
    """
    根据总热量、食物类型和用户档案，AI 智能生成营养建议（支持多用户）

    Args:
        total_calories: 总热量（千卡），可以是数字或字符串
        food_types: 食物类型列表，用逗号分隔
        username: 用户名，默认为 "default_user"
        user_context: 额外的用户上下文信息（可选）

    Returns:
        AI 生成的个性化营养评价和建议
    """
    import re
    # 处理 total_calories 可能是字符串的情况
    try:
        if isinstance(total_calories, str):
            numbers = re.findall(r'\d+', total_calories)
            if numbers:
                total_calories = int(numbers[0])
            else:
                total_calories = 0
        total_calories = int(total_calories)
    except:
        total_calories = 0

    # 读取用户档案
    profile_context = ""
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    profile_file = os.path.join("users", f"user_profile_{safe_username}.json")

    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
            goal_text = {"fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持"}.get(profile.get("goal", "maintain"), "维持")
            daily_goal = 0
            # 计算每日目标
            weight = profile.get("weight_kg", 70)
            height = profile.get("height_cm", 175)
            age = profile.get("age", 25)
            if profile.get("gender") == "male":
                bmr = int(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age))
            else:
                bmr = int(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))
            multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
            tdee = int(bmr * multipliers.get(profile.get("activity_level", "moderate"), 1.55))
            if profile.get("goal") == "fat_loss":
                daily_goal = int(tdee * 0.8)
            elif profile.get("goal") == "muscle_gain":
                daily_goal = int(tdee * 1.1)
            else:
                daily_goal = tdee

            profile_context = f"""
用户：{username}，{profile.get('age', 25)}岁，身高{profile.get('height_cm', 175)}cm，体重{profile.get('weight_kg', 70)}kg
目标：{goal_text}
每日建议热量：{daily_goal} 千卡
过敏原：{', '.join(profile.get('allergies', [])) if profile.get('allergies') else '无'}
"""
        except:
            pass

    # 构建专业的营养师 Prompt
    nutritionist_prompt = f"""你是一位专业的营养师。请根据以下信息给出营养评价和建议：

{profile_context}

当前摄入：
- 热量：{total_calories} 千卡
- 食物类型：{food_types}
{user_context}

请从以下角度给出专业分析：

1. **热量评估**
   - 这餐热量占每日建议的比例
   - 对当前目标（减脂/增肌/维持）的影响
   - 热量水平评价（偏低/适中/偏高）

2. **营养均衡性**
   - 食物类型的营养特点
   - 蛋白质/碳水/脂肪的比例评估
   - 营养缺口分析

3. **个性化建议**
   - 根据用户目标给出具体建议
   - 下一餐如何调整
   - 是否需要加餐或控制份量

4. **营养素估算（重要！）**
   - 根据食物类型和热量，估算蛋白质、碳水、脂肪含量
   - **必须在回复末尾用以下格式列出**：
     热量：{total_calories}千卡，蛋白质：XX克、碳水：XX克、脂肪：XX克

请用专业但易懂的语言回答，给出具体可操作的建议。
"""

    try:
        # 调用 LLM 生成建议
        llm = ChatOpenAI(
            model="qwen-max",
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7
        )

        response = llm.invoke(nutritionist_prompt, timeout=60)
        return response.content
    except Exception as e:
        return f"生成建议时出错：{str(e)}"

@tool
def get_user_profile(username: str = "default_user") -> str:
    """
    获取指定用户的个人档案信息（支持多用户）

    Args:
        username: 用户名，默认为 "default_user"

    Returns:
        用户的个人信息，包括身高、体重、年龄、性别、活动水平、目标和过敏原
    """
    import re
    # 将用户名转换为安全的文件名
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    user_profile_file = os.path.join("users", f"user_profile_{safe_username}.json")
    default_profile = {
        "height_cm": 175,
        "weight_kg": 70,
        "age": 25,
        "gender": "male",
        "activity_level": "moderate",
        "goal": "maintain",
        "allergies": []
    }

    try:
        if os.path.exists(user_profile_file):
            with open(user_profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)

            gender_text = "男" if profile["gender"] == "male" else "女"
            activity_text = {
                "sedentary": "久坐办公", "light": "轻度活动",
                "moderate": "中度活动", "active": "经常运动",
                "very_active": "高强度运动"
            }.get(profile["activity_level"], profile["activity_level"])
            goal_text = {
                "fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持体重"
            }.get(profile["goal"], profile["goal"])

            # 计算 BMR 和 TDEE
            weight = profile["weight_kg"]
            height = profile["height_cm"]
            age = profile["age"]
            if profile["gender"] == "male":
                bmr = int(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age))
            else:
                bmr = int(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))

            multipliers = {
                "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
                "active": 1.725, "very_active": 1.9
            }
            tdee = int(bmr * multipliers.get(profile["activity_level"], 1.55))

            if profile["goal"] == "fat_loss":
                daily_goal = int(tdee * 0.8)
            elif profile["goal"] == "muscle_gain":
                daily_goal = int(tdee * 1.1)
            else:
                daily_goal = tdee

            result = f"""【当前用户档案】
- 用户名: {username}
- 基本信息: {gender_text}, {profile['age']}岁, 身高{profile['height_cm']}cm, 体重{profile['weight_kg']}kg
- 活动水平: {activity_text}
- 健康目标: {goal_text}
- 每日建议热量: {daily_goal} 千卡
- 基础代谢(BMR): {bmr} 千卡
- 每日消耗(TDEE): {tdee} 千卡
- 过敏原: {', '.join(profile['allergies']) if profile['allergies'] else '无'}"""
            return result
        else:
            return f"用户 [{username}] 的档案文件不存在，请用户先设置个人信息。"
    except Exception as e:
        return f"读取用户档案失败: {str(e)}"


@tool
def update_user_profile(
    username: str = "default_user",
    height_cm: int = None,
    weight_kg: int = None,
    age: int = None,
    gender: str = None,
    activity_level: str = None,
    goal: str = None,
    allergies: str = None
) -> str:
    """
    更新指定用户的个人档案信息（支持多用户）

    Args:
        username: 用户名，默认为 "default_user"
        height_cm: 身高（厘米），不更新则传 None
        weight_kg: 体重（公斤），不更新则传 None
        age: 年龄，不更新则传 None
        gender: 性别（male/female），不更新则传 None
        activity_level: 活动水平（sedentary/light/moderate/active/very_active），不更新则传 None
        goal: 目标（fat_loss/muscle_gain/maintain），不更新则传 None
        allergies: 过敏原（逗号分隔的字符串），不更新则传 None

    Returns:
        更新结果确认
    """
    import re
    # 将用户名转换为安全的文件名
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    os.makedirs("users", exist_ok=True)
    user_profile_file = os.path.join("users", f"user_profile_{safe_username}.json")

    # 默认档案
    default_profile = {
        "height_cm": 175,
        "weight_kg": 70,
        "age": 25,
        "gender": "male",
        "activity_level": "moderate",
        "goal": "maintain",
        "allergies": []
    }

    # 读取现有档案
    if os.path.exists(user_profile_file):
        try:
            with open(user_profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except:
            profile = default_profile.copy()
    else:
        profile = default_profile.copy()

    # 更新指定字段
    updates = []
    if height_cm is not None and 100 <= height_cm <= 250:
        profile["height_cm"] = height_cm
        updates.append(f"身高更新为 {height_cm}cm")
    if weight_kg is not None and 30 <= weight_kg <= 200:
        profile["weight_kg"] = weight_kg
        updates.append(f"体重更新为 {weight_kg}kg")
    if age is not None and 15 <= age <= 100:
        profile["age"] = age
        updates.append(f"年龄更新为 {age}岁")
    if gender is not None and gender in ["male", "female"]:
        profile["gender"] = gender
        updates.append(f"性别更新为 {'男' if gender == 'male' else '女'}")
    if activity_level is not None and activity_level in ["sedentary", "light", "moderate", "active", "very_active"]:
        profile["activity_level"] = activity_level
        activity_names = {
            "sedentary": "久坐办公", "light": "轻度活动",
            "moderate": "中度活动", "active": "经常运动",
            "very_active": "高强度运动"
        }
        updates.append(f"活动水平更新为 {activity_names.get(activity_level, activity_level)}")
    if goal is not None and goal in ["fat_loss", "muscle_gain", "maintain"]:
        profile["goal"] = goal
        goal_names = {"fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持"}
        updates.append(f"目标更新为 {goal_names.get(goal, goal)}")
    if allergies is not None:
        profile["allergies"] = [a.strip() for a in allergies.split(",") if a.strip()]
        updates.append(f"过敏原更新为: {', '.join(profile['allergies']) if profile['allergies'] else '无'}")

    # 保存到文件
    try:
        with open(user_profile_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        if updates:
            return f"✅ 用户 [{username}] 档案已更新:\n" + "\n".join(f"- {u}" for u in updates)
        else:
            return f"ℹ️ 用户 [{username}] 没有字段需要更新"
    except Exception as e:
        return f"❌ 保存用户档案失败: {str(e)}"


@tool
def meal_decision_advice(
    current_meal_calories: int,
    meal_type: str = "lunch",
    username: str = "default_user",
    additional_context: str = ""
) -> str:
    """
    根据当前餐热量和用户档案，AI 智能给出剩餐/加餐决策建议（支持多用户）

    Args:
        current_meal_calories: 当前餐的热量（千卡）
        meal_type: 当前餐类型（breakfast/lunch/dinner/snack）
        username: 用户名，默认为 "default_user"
        additional_context: 额外上下文信息（可选）

    Returns:
        AI 生成的个性化决策建议
    """
    import re
    # 读取用户档案，计算剩余预算
    profile_context = ""
    remaining_calories = 0
    daily_goal = 0

    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    profile_file = os.path.join("users", f"user_profile_{safe_username}.json")

    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
            goal_text = {"fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持"}.get(profile.get("goal", "maintain"), "维持")

            # 计算每日目标
            weight = profile.get("weight_kg", 70)
            height = profile.get("height_cm", 175)
            age = profile.get("age", 25)
            if profile.get("gender") == "male":
                bmr = int(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age))
            else:
                bmr = int(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))
            multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
            tdee = int(bmr * multipliers.get(profile.get("activity_level", "moderate"), 1.55))
            if profile.get("goal") == "fat_loss":
                daily_goal = int(tdee * 0.8)
            elif profile.get("goal") == "muscle_gain":
                daily_goal = int(tdee * 1.1)
            else:
                daily_goal = tdee

            # 假设这是当前时间的第一餐，剩余预算 = 每日目标
            remaining_calories = daily_goal

            profile_context = f"""
用户：{username}，{profile.get('age', 25)}岁，身高{profile.get('height_cm', 175)}cm，体重{profile.get('weight_kg', 70)}kg
目标：{goal_text}
每日建议热量：{daily_goal} 千卡
当前剩余热量：{remaining_calories} 千卡
"""
        except:
            remaining_calories = 1800
            daily_goal = 1800
            profile_context = f"\n用户：{username}\n每日建议热量：{daily_goal} 千卡（默认值）\n当前剩余热量：{remaining_calories} 千卡\n"
    else:
        remaining_calories = 1800
        daily_goal = 1800
        profile_context = f"\n用户：{username}\n每日建议热量：{daily_goal} 千卡（默认值）\n当前剩余热量：{remaining_calories} 千卡\n"

    meal_names = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐", "snack": "加餐"}
    meal_cn = meal_names.get(meal_type, "这餐")

    # 构建专业的营养师 Prompt
    nutritionist_prompt = f"""你是一位专业的营养师。请根据以下信息给出用餐决策建议：

{profile_context}

当前餐食：
- 餐型：{meal_cn}
- 热量：{current_meal_calories} 千卡
- 占每日剩余比例：{(current_meal_calories / remaining_calories * 100) if remaining_calories > 0 else 0:.1f}%
{additional_context}

请给出具体的用餐决策建议：

1. **是否吃完**
   - 根据热量和用户目标判断
   - 给出明确建议：全部吃完 / 吃到七分饱 / 适当剩余 / 留到下一餐

2. **份量控制**
   - 建议摄入份量（如 1/2、2/3）
   - 优先吃哪些部分

3. **后续安排**
   - 下一餐是否需要调整
   - 是否需要加餐
   - 加餐推荐

4. **注意事项**
   - 根据用户目标的特别提醒
   - 水分摄入建议

请给出简洁、可操作的决策建议，让用户知道具体怎么做。
"""

    try:
        # 调用 LLM 生成建议
        llm = ChatOpenAI(
            model="qwen-max",
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7
        )

        response = llm.invoke(nutritionist_prompt, timeout=60)
        return response.content
    except Exception as e:
        return f"生成建议时出错：{str(e)}"


@tool
def list_user_images(username: str = "default_user") -> str:
    """
    列出指定用户图库中的所有食物图片（支持多用户）

    Args:
        username: 用户名，默认为 "default_user"

    Returns:
        用户图库中图片的列表，包含文件名和路径
    """
    import re
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    user_images_dir = os.path.join("users", f"images_{safe_username}")

    if not os.path.exists(user_images_dir):
        return f"用户 [{username}] 的图库为空，请先上传食物图片。"

    try:
        images = []
        for filename in sorted(os.listdir(user_images_dir), reverse=True):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                file_path = os.path.join(user_images_dir, filename)
                images.append({
                    "name": filename,
                    "path": file_path
                })

        if not images:
            return f"用户 [{username}] 的图库为空，请先上传食物图片。"

        # 返回图片列表，最新图片在前
        result = f"用户 [{username}] 的图库中共有 {len(images)} 张图片：\n\n"
        for i, img in enumerate(images[:10], 1):  # 最多显示10张
            result += f"{i}. {img['name']}\n   路径: {img['path']}\n"

        if len(images) > 10:
            result += f"\n...还有 {len(images) - 10} 张图片\n"

        result += "\n你可以使用 analyze_food_image 工具分析其中任何一张图片。"
        return result
    except Exception as e:
        return f"读取图库失败: {str(e)}"


@tool
def get_food_alternatives(
    food_name: str,
    current_cooking_method: str = "普通",
    username: str = "default_user",
    user_context: str = ""
) -> str:
    """
    根据用户档案和目标，AI 智能生成食物替代和烹饪方式替代建议（支持多用户）

    使用场景：用户问"可以吃什么代替"、"有什么替代"、"怎么吃更健康"时调用

    Args:
        food_name: 当前食物或菜品名称（必需）
        current_cooking_method: 当前烹饪方式（可选，默认"普通"）
        username: 用户名（可选，默认从上下文获取）
        user_context: 用户上下文信息（可选）

    Returns:
        AI 生成的个性化替代建议

    注意：如果 food_name 为空或 None，返回提示用户说明想替代什么食物
    """
    import re

    # 检查 food_name 是否有效
    if not food_name or food_name.strip() == "" or food_name == "未知食物":
        return "请告诉我您想替代哪种食物，我会为您推荐更健康的选择。"

    # 读取用户档案
    profile_context = ""
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    profile_file = os.path.join("users", f"user_profile_{safe_username}.json")

    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
            goal_text = {"fat_loss": "减脂", "muscle_gain": "增肌", "maintain": "维持"}.get(profile.get("goal", "maintain"), "维持")
            profile_context = f"""
用户：{username}，{profile.get('age', 25)}岁，身高{profile.get('height_cm', 175)}cm，体重{profile.get('weight_kg', 70)}kg
目标：{goal_text}
过敏原：{', '.join(profile.get('allergies', [])) if profile.get('allergies') else '无'}
"""
        except:
            pass

    # 构建专业的营养师 Prompt
    nutritionist_prompt = f"""你是一位专业的营养师。请根据以下信息给出食物替代建议：

{profile_context}

当前食物：{food_name}
烹饪方式：{current_cooking_method if current_cooking_method else '未知'}
{user_context}

请从以下角度给出建议：

1. **当前食物分析**
   - 营养特点（蛋白质/碳水/脂肪含量）
   - 热量水平评估
   - 是否适合用户当前目标

2. **食物替代建议**
   - 推荐2-3种更合适的替代食材
   - 说明替代理由（营养角度）
   - 考虑用户过敏原

3. **烹饪方式替代建议**
   - 推荐2-3种更健康的烹饪方式
   - 说明不同烹饪方式的营养差异
   - 给出简单可操作的做法

4. **份量建议**
   - 根据用户目标建议摄入份量
   - 如何搭配其他食物更均衡

请用专业但易懂的语言回答，格式清晰。
"""

    try:
        # 调用 LLM 生成建议
        llm = ChatOpenAI(
            model="qwen-max",
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7
        )

        response = llm.invoke(nutritionist_prompt, timeout=60)  # 增加到60秒
        return response.content
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ get_food_alternatives 错误: {error_detail}")
        return f"生成替代建议时出错：{str(e)}"


@tool
def record_meal(
    calories: int,
    protein_g: float = 0,
    carbs_g: float = 0,
    fat_g: float = 0,
    food_name: str = "未知食物",
    username: str = "default_user"
) -> str:
    """
    记录用户的饮食到今日日志（支持多用户）

    只有当用户明确表示"这是我吃的"、"刚摄入"、"已经吃了"等肯定意图时才调用此工具。
    如果用户只是问"分析一下"、"看看多少"、"帮我看看"等，则不要调用此工具。

    Args:
        calories: 热量（千卡）
        protein_g: 蛋白质（克）
        carbs_g: 碳水化合物（克）
        fat_g: 脂肪（克）
        food_name: 食物名称描述
        username: 用户名，默认为 "default_user"

    Returns:
        记录确认信息
    """
    import re
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    meal_history_file = os.path.join("users", f"meal_history_{safe_username}.json")

    # 读取现有记录
    meal_history = []
    if os.path.exists(meal_history_file):
        try:
            with open(meal_history_file, "r", encoding="utf-8") as f:
                meal_history = json.load(f)
        except:
            pass

    # 添加新记录
    from datetime import datetime
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "foods": [food_name],
        "total_calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "image_path": ""
    }
    meal_history.append(record)

    # 保存
    os.makedirs("users", exist_ok=True)
    try:
        with open(meal_history_file, "w", encoding="utf-8") as f:
            json.dump(meal_history, f, ensure_ascii=False, indent=2)

        # 计算今日总计
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = [r for r in meal_history if r["date"].startswith(today)]
        total_calories = sum(r["total_calories"] for r in today_records)
        total_protein = sum(r["protein_g"] for r in today_records)
        total_carbs = sum(r["carbs_g"] for r in today_records)
        total_fat = sum(r["fat_g"] for r in today_records)

        return f"""✅ 已记录到今日饮食：
- 热量：{calories}千卡
- 蛋白质：{protein_g}克
- 碳水：{carbs_g}克
- 脂肪：{fat_g}克

📊 今日累计：
- 总热量：{total_calories}千卡
- 蛋白质：{total_protein:.0f}克
- 碳水：{total_carbs:.0f}克
- 脂肪：{total_fat:.0f}克
- 餐数：{len(today_records)}餐"""
    except Exception as e:
        return f"❌ 记录失败: {str(e)}"


@tool
def undo_last_meal_record(username: str = "default_user") -> str:
    """
    撤销上一条饮食记录（支持多用户）

    当用户说"取消记录"、"刚才不算"、"删除记录"、"撤回"等时调用此工具。

    Args:
        username: 用户名，默认为 "default_user"

    Returns:
        撤销结果
    """
    import re
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    meal_history_file = os.path.join("users", f"meal_history_{safe_username}.json")

    if not os.path.exists(meal_history_file):
        return "⚠️ 没有找到饮食记录"

    try:
        with open(meal_history_file, "r", encoding="utf-8") as f:
            meal_history = json.load(f)

        if not meal_history:
            return "⚠️ 没有可撤销的记录"

        # 删除最后一条
        removed = meal_history.pop()

        # 保存
        with open(meal_history_file, "w", encoding="utf-8") as f:
            json.dump(meal_history, f, ensure_ascii=False, indent=2)

        return f"🗑️ 已撤销上一条记录：{removed['total_calories']}千卡"
    except Exception as e:
        return f"❌ 撤销失败: {str(e)}"


@tool
def get_today_meal_records(username: str = "default_user") -> str:
    """
    获取今日所有的饮食记录（支持多用户）

    当用户要求查看今日记录、或需要修正记录时调用此工具。

    Args:
        username: 用户名，默认为 "default_user"

    Returns:
        今日所有饮食记录的详细信息
    """
    import re
    from datetime import datetime
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    meal_history_file = os.path.join("users", f"meal_history_{safe_username}.json")

    if not os.path.exists(meal_history_file):
        return "今日还没有饮食记录。"

    try:
        with open(meal_history_file, "r", encoding="utf-8") as f:
            meal_history = json.load(f)

        # 筛选今日记录
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = [r for r in meal_history if r["date"].startswith(today)]

        if not today_records:
            return "今日还没有饮食记录。"

        result = f"今日共有 {len(today_records)} 条记录：\n\n"
        for i, record in enumerate(today_records, 1):
            foods = ", ".join(record.get("foods", ["未知"]))
            result += f"{i}. {record['date'].split(' ')[1]} - {foods}\n"
            result += f"   热量：{record['total_calories']}千卡\n"
            result += f"   蛋白质：{record['protein_g']}g，碳水：{record['carbs_g']}g，脂肪：{record['fat_g']}g\n\n"

        # 计算总计
        total_calories = sum(r["total_calories"] for r in today_records)
        result += f"今日总计：{total_calories}千卡"

        return result
    except Exception as e:
        return f"读取记录失败: {str(e)}"


@tool
def clear_today_meals(username: str = "default_user") -> str:
    """
    清空今日所有饮食记录（支持多用户）

    当用户明确要求"重新记录"、"清空今日"、"今日记录错了"等时调用此工具。

    Args:
        username: 用户名，默认为 "default_user"

    Returns:
        清空结果
    """
    import re
    from datetime import datetime
    safe_username = re.sub(r'[^\w\u4e00-\u9fff]', '_', str(username))
    meal_history_file = os.path.join("users", f"meal_history_{safe_username}.json")

    if not os.path.exists(meal_history_file):
        return "今日还没有饮食记录。"

    try:
        with open(meal_history_file, "r", encoding="utf-8") as f:
            meal_history = json.load(f)

        # 筛选出非今日的记录
        today = datetime.now().strftime("%Y-%m-%d")
        non_today_records = [r for r in meal_history if not r["date"].startswith(today)]

        # 保存（只保留非今日记录）
        os.makedirs("users", exist_ok=True)
        with open(meal_history_file, "w", encoding="utf-8") as f:
            json.dump(non_today_records, f, ensure_ascii=False, indent=2)

        today_count = len(meal_history) - len(non_today_records)
        return f"✅ 已清空今日 {today_count} 条记录。今日饮食记录已重置。"
    except Exception as e:
        return f"❌ 清空失败: {str(e)}"
    except Exception as e:
        return f"❌ 撤销失败: {str(e)}"


# 工具列表
tools = [
    analyze_food_image,
    calculate_total_calories,
    get_nutrition_advice,
    meal_decision_advice,
    get_user_profile,
    update_user_profile,
    get_food_alternatives,
    list_user_images,
    record_meal,
    undo_last_meal_record,
    get_today_meal_records,
    clear_today_meals
]
tools_by_name = {t.name: t for t in tools}

# ==============================================
# 【状态定义】LangGraph State
# ==============================================

class AgentState(dict):
    """
    Agent 状态
    - messages: 完整的对话历史（LangChain 标准格式）
    - image_analyzed: 当前图片是否已分析
    - image_info: 图片分析结果（结构化信息）
    """
    messages: Annotated[list[BaseMessage], add_messages]
    image_analyzed: bool
    image_info: dict


# ==============================================
# 【节点1】视觉分析节点 - Qwen-VL
# ==============================================

def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # 压缩图片以加快处理
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def vision_analysis_node(state: AgentState) -> dict:
    """
    使用 Qwen-VL 分析图片，提取结构化的食材和营养信息
    
    返回：
    - image_info: 包含食材列表、估算热量的字典
    - image_analyzed: True
    """
    print("\n" + "="*60)
    print("📷 视觉分析节点 - Qwen-VL")
    print("="*60)
    
    # 获取最新的用户消息
    last_message = state["messages"][-1]
    
    # 检查是否有图片
    image_base64 = None
    user_question = ""
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        image_base64 = item["image_url"]["url"].replace("data:image/jpeg;base64,", "")
                    elif item.get("type") == "text":
                        user_question = item["text"]
        elif isinstance(content, str):
            user_question = content
    
    if not image_base64:
        # 没有图片，跳过此节点
        print("ℹ️  无图片，跳过视觉分析")
        return {"image_analyzed": True, "image_info": {}}
    
    print("🔄 正在分析图片...")
    
    # 配置 Qwen-VL
    dashscope.api_key = DASHSCOPE_API_KEY
    
    # 构建 Prompt - 要求返回结构化数据
    prompt = f"""请分析这张图片中的食物，并以 JSON 格式返回以下信息：

{{
    "foods": [
        {{
            "name": "食物名称",
            "estimated_weight_g": 估算重量(克),
            "calories_per_100g": 每100克热量(千卡),
            "estimated_calories": 估算热量(千卡),
            "protein_g": 蛋白质含量(克),
            "carbs_g": 碳水化合物含量(克),
            "fat_g": 脂肪含量(克)
        }}
    ],
    "total_estimated_calories": 总热量估算(千卡),
    "total_protein_g": 总蛋白质(克),
    "total_carbs_g": 总碳水(克),
    "total_fat_g": 总脂肪(克),
    "nutrition_tags": ["高蛋白", "低脂", "高纤维"等标签]
}}

要求：
1. 尽可能准确地识别所有食材
2. 基于常见份量估算重量
3. 使用标准的营养数据库值，估算蛋白质、碳水、脂肪含量
4. 必须返回完整的营养素信息（蛋白质、碳水、脂肪）
5. 只返回 JSON，不要有其他文字

用户问题：{user_question if user_question else "分析图片中的食物"}
"""
    
    messages = [
        {
            'role': 'system',
            'content': [{'text': '你是专业的营养师助手，擅长食物识别和营养分析'}]
        },
        {
            'role': 'user',
            'content': [
                {'image': f"data:image/jpeg;base64,{image_base64}"},
                {'text': prompt}
            ]
        }
    ]
    
    # 调用 Qwen-VL
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-max',
        messages=messages
    )
    
    if response.status_code != 200:
        print(f"❌ 视觉分析失败: {response.code} - {response.message}")
        return {"image_analyzed": True, "image_info": {}}
    
    # 解析结果
    try:
        result_text = response.output.choices[0].message.content[0]['text']
        print("✅ 图片分析完成")
        print(f"📊 分析结果预览: {result_text[:200]}...")
        
        # 尝试解析 JSON（这里简化处理，实际应该用 json.loads）
        image_info = {
            "raw_analysis": result_text,
            "question": user_question
        }
        
        return {
            "image_analyzed": True,
            "image_info": image_info
        }
    except Exception as e:
        print(f"❌ 解析分析结果失败: {e}")
        return {"image_analyzed": True, "image_info": {}}


# ==============================================
# 【节点2】推理节点 - Qwen-Max
# ==============================================

def reasoning_node(state: AgentState) -> dict:
    """
    使用 Qwen-Max 进行 ReAct 推理和对话（纯文本模型）
    
    功能：
    - 理解用户意图
    - 决定是否需要调用工具
    - 生成最终回复
    """
    print("\n" + "="*60)
    print("🧠 推理节点 - Qwen-Max")
    print("="*60)
    
    # 初始化 Qwen-Max（带工具）- 使用 OpenAI 兼容接口
    llm = ChatOpenAI(
        model="qwen-max",  # 使用纯文本推理模型
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # OpenAI 兼容接口
        temperature=0.7
    ).bind_tools(tools)
    
    # 构建系统提示
    system_prompt = """你是专业的营养师助手。你有以下工具可以帮助用户：

**可用工具：**
1. analyze_food_image - 分析食物图片
2. calculate_total_calories - 计算总热量
3. get_nutrition_advice - 生成营养建议
4. meal_decision_advice - 生成用餐决策建议
5. get_user_profile - 获取用户档案
6. update_user_profile - 更新用户档案
7. get_food_alternatives - 推荐食物替代
8. list_user_images - 列出用户图库中的图片
9. record_meal - 记录饮食到今日日志
10. undo_last_meal_record - 撤销上一条记录
11. get_today_meal_records - 查看今日所有记录
12. clear_today_meals - 清空今日所有记录

**用户的消息中包含用户档案和今日摄入情况，请注意查看！**

**你的工作方式：**
1. 理解用户的意图和需求
2. 根据需要调用合适的工具
3. 综合工具结果，给出专业、友好的回复

**关于记录饮食：**
- 只有当用户明确表示这餐是他们吃的时候，才调用 record_meal
- 如果用户只是询问、分析，不要记录
- 如果用户说记错了、要重新记录，调用 clear_today_meals 清空后重新记录

**回复格式：**
- 自然、专业、友好
- 分析食物时列出营养素：热量、蛋白质、碳水、脂肪
- 不要解释工具调用过程
"""
    
    # 构建消息列表（系统提示 + 对话历史）
    messages_with_context = [SystemMessage(content=system_prompt)]
    
    # 添加对话历史
    messages_with_context.extend(state["messages"])
    
    print("💭 正在思考...")
    
    # 调用 LLM（增加超时时间）
    try:
        response = llm.invoke(
            messages_with_context,
            timeout=60  # 60秒超时（原来是30秒，减少超时错误）
        )
        print("✅ 推理完成")
    except Exception as e:
        error_msg_str = str(e)
        print(f"❌ 推理失败: {error_msg_str}")
        
        # 检查是否是工具调用相关的错误
        if "tool" in error_msg_str.lower() or "InvalidParameter" in error_msg_str:
            print("\n⚠️  检测到工具调用格式错误，尝试恢复...")
            # 移除最后一条可能有问题的消息，重新尝试
            if len(state["messages"]) > 0:
                # 返回一个引导性的消息，让对话继续
                recovery_msg = AIMessage(content="抱歉，我在处理工具调用时遇到了一些问题。让我重新为您分析。\n\n您可以告诉我具体的食物和份量，我会帮您计算热量。")
                return {"messages": [recovery_msg]}
        
        # 其他错误
        print(f"\n提示：请检查 API 配置和网络连接")
        error_msg = AIMessage(content=f"抱歉，我遇到了一些问题：{str(e)[:200]}\n\n请稍后重试或联系管理员。")
        return {"messages": [error_msg]}
    
    # 返回更新的消息列表
    return {"messages": [response]}


# ==============================================
# 【路由函数】决定下一步
# ==============================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    判断是否需要调用工具
    
    检查最后一条 AI 消息是否有工具调用
    """
    last_message = state["messages"][-1]
    
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("🔧 检测到工具调用，执行工具节点")
        return "tools"
    
    print("✨ 无需工具，直接结束")
    return "end"


# ==============================================
# 【自定义工具节点】处理 Qwen 的工具调用
# ==============================================

def custom_tool_node(state: AgentState) -> dict:
    """
    自定义工具执行节点，更好地处理 Qwen 的工具调用
    """
    print("\n" + "="*60)
    print("🔧 工具执行节点")
    print("="*60)
    
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("⚠️  没有检测到工具调用")
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    print(f"📋 工具调用数量: {len(tool_calls)}")
    
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")
        
        print(f"\n🛠️  执行工具: {tool_name}")
        print(f"📝 参数: {tool_args}")
        
        # 查找对应的工具
        if tool_name in tools_by_name:
            try:
                tool_func = tools_by_name[tool_name]
                result = tool_func.invoke(tool_args)
                print(f"✅ 工具执行成功: {result[:100]}..." if len(str(result)) > 100 else f"✅ 工具执行成功: {result}")
                
                # 创建工具响应消息（Qwen 要求的格式）
                from langchain_core.messages import ToolMessage
                tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_msg)
                
            except Exception as e:
                print(f"❌ 工具执行失败: {e}")
                from langchain_core.messages import ToolMessage
                error_msg = ToolMessage(
                    content=f"工具执行错误: {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)
        else:
            print(f"❌ 未找到工具: {tool_name}")
            from langchain_core.messages import ToolMessage
            error_msg = ToolMessage(
                content=f"未知工具: {tool_name}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(error_msg)
    
    print(f"\n✅ 共执行 {len(tool_messages)} 个工具")
    return {"messages": tool_messages}


# ==============================================
# 【构建工作流】
# ==============================================

def build_nutritionist_agent():
    """
    构建营养师 Agent 工作流
    
    流程：
    1. 推理节点（始终执行）- 自主决定是否需要调用工具
    2. 判断是否需要工具 → 工具节点或直接结束
    3. 工具执行后回到推理节点
    
    注意：视觉分析现在是 analyze_food_image 工具，由推理节点自主调用
    """
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tools", custom_tool_node)  # 使用自定义工具节点
    
    # 设置入口点 - 直接进入推理节点
    workflow.set_entry_point("reasoning")
    
    # 推理后的条件路由
    workflow.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # 工具执行后回到推理
    workflow.add_edge("tools", "reasoning")
    
    # 编译工作流（带记忆）
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    
    return agent


# ==============================================
# 【使用示例】
# ==============================================

if __name__ == "__main__":
    # 检查 API Keys
    print("🔍 检查 API 配置...")
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "":
        print("❌ 错误: DASHSCOPE_API_KEY 未配置")
        exit(1)
    else:
        print(f"✅ Qwen API Key: {DASHSCOPE_API_KEY[:15]}...")
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        # 使用 Qwen-Max 作为推理模型（纯文本）
        print("ℹ️  使用 Qwen-Max 作为推理模型（纯文本）")
    else:
        print(f"✅ OpenAI API Key: {OPENAI_API_KEY[:15]}...")
    
    print("\n" + "="*60)
    
    # 创建 Agent
    agent = build_nutritionist_agent()
    
    # 配置线程 ID（用于多轮对话记忆）
    thread_id = "nutritionist_chat_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("="*60)
    print("🥗 营养师助手 - 全新架构")
    print("="*60)
    print(f"💾 对话记忆已启用 (thread_id: {thread_id})")
    print("="*60)
    
    # ========== 测试场景1：带图片的对话（使用视觉工具） ==========
    print("\n\n📝 测试场景1：分析食物图片")
    print("-"*60)
    
    # 准备图片
    image_path = "test.jpg"
    if os.path.exists(image_path):
        initial_state = {
            "messages": [
                HumanMessage(content=f"请帮我分析这张图片的热量和营养：{image_path}")
            ],
            "image_analyzed": False,
            "image_info": {}
        }
        
        # 运行
        result = agent.invoke(initial_state, config=config)
        
        # 显示最终回复
        print("\n" + "="*60)
        print("📊 最终回复：")
        print("="*60)
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                print(msg.content)
    else:
        print(f"⚠️  测试图片不存在: {image_path}")
        print("💡 提示：请准备一张 test.jpg 图片进行测试")
    
    # ========== 测试场景2：纯文本的第二轮对话（有记忆）==========
    print("\n\n📝 测试场景2：追问（测试多轮对话记忆）")
    print("-"*60)
    
    follow_up_state = {
        "messages": [
            HumanMessage(content="那这个热量算高吗？我应该怎么调整？")
        ],
        "image_analyzed": True,  # 已经分析过了
        "image_info": {}
    }
    
    result2 = agent.invoke(follow_up_state, config=config)
    
    print("\n" + "="*60)
    print("📊 追问回复：")
    print("="*60)
    for msg in result2["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)