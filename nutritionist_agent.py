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
def analyze_food_image(image_path: str, analysis_instruction: str = "请分析这张图片中的食物，并以 JSON 格式返回以下信息：\n\n{\n    \"foods\": [\n        {\n            \"name\": \"食物名称\",\n            \"estimated_weight_g\": 估算重量(克),\n            \"calories_per_100g\": 每100克热量(千卡),\n            \"estimated_calories\": 估算热量(千卡)\n        }\n    ],\n    \"total_estimated_calories\": 总热量估算,\n    \"nutrition_tags\": [\"高蛋白\", \"低脂\", \"高纤维\"等标签]\n}\n\n要求：\n1. 尽可能准确地识别所有食材\n2. 基于常见份量估算重量\n3. 使用标准的营养数据库值\n4. 只返回 JSON，不要有其他文字") -> str:
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
        calories_list: 热量列表，格式为 "数字1+数字2+数字3"，例如 "100+200+50"
    
    Returns:
        总热量数值
    """
    try:
        # 清理输入，只保留数字和运算符
        import re
        cleaned = re.findall(r'[\d\+\-\*\/\.]+', calories_list)[0]
        result = eval(cleaned)
        return f"总热量：{result} 千卡"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def get_nutrition_advice(total_calories: int, food_types: str) -> str:
    """
    根据总热量和食物类型提供营养建议
    
    Args:
        total_calories: 总热量（千卡）
        food_types: 食物类型列表，用逗号分隔
    
    Returns:
        营养评价和建议
    """
    advice = f"基于 {total_calories} 千卡的摄入，包含 {food_types}\n\n"
    
    if total_calories < 400:
        advice += "✅ 低热量餐，适合减脂期\n"
    elif total_calories < 700:
        advice += "✅ 适中热量，营养均衡\n"
    else:
        advice += "⚠️ 较高热量，注意控制份量\n"
    
    advice += "\n建议：保持多样化饮食，确保蛋白质、碳水、脂肪的合理搭配。"
    return advice


# 工具列表
tools = [analyze_food_image, calculate_total_calories, get_nutrition_advice]
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
            "estimated_calories": 估算热量(千卡)
        }}
    ],
    "total_estimated_calories": 总热量估算,
    "nutrition_tags": ["高蛋白", "低脂", "高纤维"等标签]
}}

要求：
1. 尽可能准确地识别所有食材
2. 基于常见份量估算重量
3. 使用标准的营养数据库值
4. 只返回 JSON，不要有其他文字

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
    system_prompt = """你是专业的营养师助手，具备以下能力：

1. **食物分析**：使用 analyze_food_image 工具识别图片中的食物
2. **热量计算**：使用 calculate_total_calories 工具精确计算总热量
3. **营养建议**：使用 get_nutrition_advice 工具提供科学的饮食建议
4. **多轮对话**：记住之前的对话内容

【工作流程】
- 如果用户提供了图片文件路径（例如 "test.jpg"），调用 analyze_food_image 工具
  * 你可以根据需要自定义 analysis_instruction 参数，告诉模型如何分析
  * 默认情况下，模型会返回标准的营养成分 JSON
  * 示例：
    - 标准分析：使用默认参数
    - 只识别食材：analysis_instruction="请列出图片中的所有食材名称"
    - 重点分析蛋白质：analysis_instruction="请重点分析这顿饭的蛋白质含量和来源"
    - 减脂评估：analysis_instruction="判断这顿饭是否适合减脂期食用，并说明原因"
- 如果需要计算总热量，调用 calculate_total_calories 工具
- 如果需要营养建议，调用 get_nutrition_advice 工具
- 保持对话自然流畅

【重要】
- 只在确实需要时才调用工具
- 调用工具时提供准确的参数
- 收到工具结果后，给出友好的回答
- 当用户提到图片时，询问图片的文件路径
- 根据用户需求灵活定制分析指令
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
