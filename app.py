"""
营养师助手 - Streamlit 版本
运行：streamlit run app_streamlit.py
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from nutritionist_agent import build_nutritionist_agent
import streamlit as st
from PIL import Image

load_dotenv()

# ==============================================
# 初始化 Agent（只加载一次）
# ==============================================
@st.cache_resource
def get_agent():
    """缓存 Agent，避免重复加载"""
    return build_nutritionist_agent()

def get_final_ai_response(result_messages):
    """提取最终 AI 回复"""
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "抱歉，我暂时无法回复。"

# ==============================================
# 页面配置
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
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">🥗 营养师助手</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">上传食物图片，AI 帮你识别食材、计算热量、给出营养建议</p>', unsafe_allow_html=True)

# ==============================================
# 侧边栏：图片上传和说明
# ==============================================
with st.sidebar:
    st.markdown("### 📸 上传食物图片")
    
    uploaded_file = st.file_uploader(
        "点击或拖拽上传图片",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # 保存图片路径
    if uploaded_file is not None:
        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "food_image.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="已上传图片", use_container_width=True)
        st.success(f"✅ 图片已上传")
    else:
        image_path = None
        st.info("⚡ 等待上传图片")
    
    st.markdown("---")
    st.markdown("### 💡 试试这些问题")
    st.markdown("""
    - 分析这张图片
    - 这餐热量高吗？
    - 适合减脂期吃吗？
    - 蛋白质够不够？
    - 有什么建议吗？
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ 使用说明")
    st.markdown("""
    1. 上传食物图片（可选）
    2. 输入你的问题
    3. 等待 AI 回复
    """)
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================
# 初始化会话状态
# ==============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "image_analyzed": False,
        "image_info": {}
    }

# ==============================================
# 显示历史消息
# ==============================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🥗"):
            st.markdown(msg["content"])

# ==============================================
# 处理用户输入
# ==============================================
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 构建完整消息
    if image_path:
        full_message = f"{prompt}\n{image_path}"
    else:
        full_message = prompt
    
    # 调用 Agent
    st.session_state.agent_state["messages"] = [HumanMessage(content=full_message)]
    
    with st.chat_message("assistant", avatar="🥗"):
        with st.spinner("🤔 思考中..."):
            try:
                agent = get_agent()
                thread_id = "streamlit_chat"
                config = {"configurable": {"thread_id": thread_id}}
                
                result = agent.invoke(st.session_state.agent_state, config=config)
                bot_response = get_final_ai_response(result["messages"])
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.session_state.agent_state.update(result)
                
            except Exception as e:
                error_msg = f"❌ 出错啦：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})