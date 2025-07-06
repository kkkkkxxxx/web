import streamlit as st 
import os
from datetime import datetime
from fact_checker_v4_english import FactChecker
import time

# 页面配置
st.set_page_config(
    page_title="AI虚假新闻检测器",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 应用标题和描述
st.title("AI虚假新闻检测器")
st.markdown("""
本应用程序使用API调用LLM验证陈述的准确性。
请在下方输入需要核查的新闻，系统将检索网络证据进行新闻核查，无网络时将只进行本地知识库检索。
""")

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    
    # 本地API端点设置
    api_base = st.text_input("API基础URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                             help="您的本地API端点的基础URL")
    
    # 模型选择 - 已更新为本地Qwen模型
    model_option = st.selectbox(
        "选择模型",
        ["qwen2.5-14b-instruct-1m"],  # 你可以扩展更多模型选项
        index=0,
        help="使用本地Qwen2.5模型"
    )
    
    # 高级设置折叠部分
    with st.expander("高级设置"):
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.0, step=0.1, 
                               help="较低的值使响应更确定，较高的值使响应更具创造性")
        max_tokens = st.slider("最大响应长度", min_value=100, max_value=8000, value=1000, step=100,
                              help="响应中的最大标记数")
    
    st.divider()
    st.markdown("### 关于 ###")
    st.markdown("虚假新闻检测器:")
    st.markdown("1. 从新闻中提取核心声明")
    st.markdown("2. 在网络上和本地库搜索证据")
    st.markdown("3. 使用BGE-M3按相关性对证据进行排名")
    st.markdown("4. 基于证据提供结论")
    st.markdown("使用LLM、Streamlit、BGE-M3和rag开发 ❤️❤️❤️")

# 如果不存在，初始化会话状态以存储聊天历史
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 主输入区域
user_input = st.chat_input("请在下方输入需要核查的新闻...")

if user_input:
    # 将用户消息添加到聊天历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 使用 with 上下文创建助手消息容器
    with st.chat_message("assistant"):
        # 创建多个独立占位符，避免 DOM 错误
        claim_placeholder = st.empty()
        information_placeholder = st.empty()
        evidence_placeholder = st.empty()
        verdict_placeholder = st.empty()

        # 初始化FactChecker
        fact_checker = FactChecker(api_base, model_option, temperature, max_tokens)

        # 第1步：提取声明
        claim_placeholder.markdown("### 🔍 正在提取新闻的核心声明...")
        claim = fact_checker.extract_claim(user_input)

        # 处理claim字符串，提取"claim:"后面的内容
        if "claim:" in claim.lower():
            claim = claim.split("claim:")[-1].strip()
        claim_placeholder.markdown(f"### 🔍 提取新闻的核心声明\n\n{claim}")

        # 提取关键信息
        information_placeholder.markdown("### 🔍 正在提取新闻的关键信息...")
        information = fact_checker.extract_keyinformation(user_input)
        information_placeholder.markdown(f"### 🔍 提取新闻的关键信息\n\n{information}")

        # 第2步：搜索证据
        evidence_placeholder.markdown("### 🌐 正在搜索相关证据...")
        evidence_docs = fact_checker.search_evidence(claim)

        # 第3步：获取相关证据块
        evidence_placeholder.markdown("### 🌐 正在分析证据相关性...")
        evidence_chunks = fact_checker.get_evidence_chunks(evidence_docs, claim)

        # 显示证据结果
        evidence_md = "### 🔗 证据来源\n\n"
        for j, chunk in enumerate(evidence_chunks[:-1]):  # 跳过最后一个，与原始代码保持一致
            evidence_md += f"**[{j+1}]:**\n"
            evidence_md += f"{chunk['text']}\n"
            evidence_md += f"来源: {chunk['source']}\n\n"
        evidence_placeholder.markdown(evidence_md)

        # 第4步：评估声明
        verdict_placeholder.markdown("### ⚖️ 正在评估声明真实性...")
        evaluation = fact_checker.evaluate_claim(information, user_input, evidence_chunks)

        # 确定结论表情符号
        verdict = evaluation["verdict"]
        if verdict.upper() == "TRUE":
            emoji = "✅"
            verdict_cn = "正确"
        elif verdict.upper() == "FALSE":
            emoji = "❌"
            verdict_cn = "错误"
        elif verdict.upper() == "PARTIALLY TRUE":
            emoji = "⚠️"
            verdict_cn = "部分正确"
        else:
            emoji = "❓"
            verdict_cn = "无法验证"

        # 显示最终结论
        verdict_md = f"### {emoji} 结论: {verdict_cn}\n\n"
        verdict_md += f"### 推理过程\n\n{evaluation['reasoning']}\n\n"

        verdict_placeholder.markdown(verdict_md)

        # 整合完整的响应内容用于保存到聊天历史
        full_response = f"""
### 🔍 提取新闻的核心声明

{claim}

---

{information}

---

{evidence_md}

---

{verdict_md}
"""

        # 添加助手响应到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})
