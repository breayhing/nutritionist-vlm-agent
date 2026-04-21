# 🥗 营养师助手 AI Agent

基于 LangGraph + Qwen 双模型架构的智能营养师助手，支持图片识别、热量计算、营养建议和多轮对话。

---

## ✨ 核心特性

- **双模型架构**：Qwen-VL-Max（视觉分析）+ Qwen-Max（推理对话）
- **智能工具调用**：Agent 自主决定何时调用什么工具
- **自定义分析指令**：根据用户需求灵活调整分析重点
- **多轮对话记忆**：支持连续对话，记住上下文
- **图片随时发送**：不必在首轮发送图片，任何轮次都可以

---

## 📦 安装依赖

```bash
pip install langchain langgraph langchain-openai dashscope python-dotenv pillow
```

**必需的环境变量：**

在项目根目录创建 `.env` 文件：
```env
DASHSCOPE_API_KEY=你的阿里云DashScope API密钥
```

获取 API Key：[阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)

---

## 🚀 快速开始

### 1. 准备图片

将食物图片放到 `project/` 目录，例如：`my_lunch.jpg`

> ⚠️ 注意：图片不要放在 `test/` 文件夹

### 2. 运行程序

```bash
cd project
python chat_demo.py
```

### 3. 开始对话

```
👤 你：分析 my_lunch.jpg
🤖 AI：这顿饭包含鸡胸肉、西兰花...总热量约 500 千卡

👤 你：这个热量算高吗？
🤖 AI：基于之前的分析，500 千卡属于适中范围...

👤 你：quit
```

---

## 📖 用户使用说明

### 基本用法

**分析图片：**
```
分析 breakfast.jpg
```

**自定义分析需求：**
```
帮我看看 lunch.png 里有哪些食材，不需要热量
分析 dinner.jpg，重点看蛋白质含量够不够
这张图适合减脂期吃吗：snack.jpg
```

**纯文本对话：**
```
我想了解健康午餐怎么搭配
每天应该摄入多少蛋白质？
```

**多轮对话（有记忆）：**
```
第1轮：分析 my_food.jpg
第2轮：那这个热量算高吗？        ← AI 记得第1轮的分析
第3轮：我应该怎么调整？           ← AI 继续基于上下文给出建议
```

### 支持的场景

- ✅ 标准营养分析
- ✅ 只识别食材名称
- ✅ 重点分析某营养素（蛋白质、脂肪等）
- ✅ 减脂/增肌评估
- ✅ 过敏原检测
- ✅ 儿童营养评估
- ✅ 食材替代建议

---

## 👨‍💻 开发者二次开发说明

### 项目结构

```
project/
├── nutritionist_agent.py      # 核心代码（Agent 实现）
├── chat_demo.py               # 用户交互入口
├── test/                      # 测试文件
│   ├── test_tool_call.py
│   ├── test_quick.py
│   ├── test_comprehensive.py
│   └── test_custom_vision_instruction.py
├── .env                       # API 配置
└── *.jpg                      # 示例图片
```

### 架构设计

**双模型策略：**
- **Qwen-VL-Max**：负责视觉分析（通过 `analyze_food_image` 工具）
- **Qwen-Max**：负责推理和对话（推理节点）

**工作流程：**
```
用户输入 → 推理节点（Qwen-Max）→ 判断是否需要工具
                                    ↓
                            是 → 工具节点 → 回到推理节点
                                    ↓
                            否 → 直接回复用户
```

**三大工具：**
1. `analyze_food_image` - 视觉分析（支持自定义指令）
2. `calculate_total_calories` - 热量计算
3. `get_nutrition_advice` - 营养建议

### 添加新工具

**步骤1：定义工具函数**

在 `nutritionist_agent.py` 中添加：

```python
@tool
def your_new_tool(param1: str, param2: int = 0) -> str:
    """
    工具描述（重要！模型会根据这个决定何时调用）
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    """
    # 你的逻辑
    return result
```

**步骤2：注册工具**

找到工具列表部分（约第 169 行）：

```python
tools = [analyze_food_image, calculate_total_calories, get_nutrition_advice, your_new_tool]
tools_by_name = {t.name: t for t in tools}
```

**步骤3：更新系统提示词**

在 `reasoning_node` 函数的 `system_prompt` 中添加工具使用说明。

**示例：添加食物相克检测工具**

```python
@tool
def check_food_compatibility(food_list: str) -> str:
    """
    检查食物之间是否存在相克或不推荐的搭配
    
    Args:
        food_list: 食物列表，用逗号分隔，例如 "螃蟹,柿子,啤酒"
    
    Returns:
        相克检测结果和建议
    """
    foods = [f.strip() for f in food_list.split(",")]
    # 这里可以接入专业知识库或 API
    return f"检测到 {len(foods)} 种食物，未发现明显相克..."

# 注册工具
tools.append(check_food_compatibility)
tools_by_name["check_food_compatibility"] = check_food_compatibility
```

### 修改分析指令模板

如果想改变默认的视觉分析格式，修改 `analyze_food_image` 工具的默认参数：

```python
@tool
def analyze_food_image(
    image_path: str, 
    analysis_instruction: str = "你的自定义默认指令..."
) -> str:
```

### 调试技巧

**查看工具调用过程：**
运行时会输出详细信息：
```
🔍 [视觉工具] 正在分析图片: xxx.jpg
📝 [视觉工具] 分析指令: ...
✅ [视觉工具] 分析完成
📊 [视觉工具] 结果预览: ...
```

**快速验证代码：**
```bash
cd test
python test_quick.py
```

**超时问题优化：**
- 增加 `timeout` 参数（当前 60 秒）
- 压缩图片尺寸
- 检查网络连接

---

## 🔧 常见问题

**Q: 为什么会出现超时错误？**
A: 可能原因：网络不稳定、API 繁忙、图片太大。建议检查网络或稍后重试。

**Q: 如何更换测试图片？**
A: 将图片放到 `project/` 目录，在对话中指定路径即可，无需修改代码。

**Q: 多轮对话没有记忆？**
A: 确保使用相同的 `thread_id`，Checkpointer 会自动保存状态。

**Q: 模型没有调用工具？**
A: 确保明确提到图片路径，且问题与食物/营养相关。

---

## 📝 技术栈

- **框架**：LangGraph（工作流编排）
- **模型**：Qwen-VL-Max（视觉）、Qwen-Max（推理）
- **工具**：LangChain Tools
- **记忆**：MemorySaver（对话历史持久化）

---

## 📄 License

本项目仅供学习和个人使用。

---

## 🙏 致谢

- 阿里云 DashScope API
- LangChain & LangGraph 团队
