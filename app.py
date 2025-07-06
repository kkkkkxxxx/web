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

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .description {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
        background-color: #e8f4f8;
    }
    
    .evidence-item {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #28a745;
    }
    
    .verdict-true {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .verdict-false {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .verdict-partial {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .verdict-unknown {
        background-color: #e2e3e5;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题和描述
st.markdown('<h1 class="main-header">🔍 AI虚假新闻检测器</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="description">
    <p>本应用程序使用API调用LLM验证陈述的准确性。</p>
    <p>请在下方输入需要核查的新闻，系统将检索网络证据进行新闻核查，无网络时将只进行本地知识库检索。</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("⚙️ 配置")
    
    # 本地API端点设置
    api_base = st.text_input(
        "API基础URL", 
        value="https://dashscope.aliyuncs.com/compatible-mode/v1", 
        help="您的本地API端点的基础URL"
    )
    
    # 模型选择
    model_option = st.selectbox(
        "选择模型",
        ["qwen2.5-14b-instruct-1m"],
        index=0,
        help="使用本地Qwen2.5模型"
    )
    
    # 高级设置折叠部分
    with st.expander("🔧 高级设置"):
        temperature = st.slider(
            "温度", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.1, 
            help="较低的值使响应更确定，较高的值使响应更具创造性"
        )
        max_tokens = st.slider(
            "最大响应长度", 
            min_value=100, 
            max_value=8000, 
            value=1000, 
            step=100,
            help="响应中的最大标记数"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # 关于部分
    st.markdown("### 📖 关于")
    st.markdown("**虚假新闻检测器工作流程:**")
    st.markdown("1. 🎯 从新闻中提取核心声明")
    st.markdown("2. 🔍 在网络上和本地库搜索证据")
    st.markdown("3. 📊 使用BGE-M3按相关性对证据进行排名")
    st.markdown("4. ⚖️ 基于证据提供结论")
    st.markdown("---")
    st.markdown("**技术栈:** LLM + Streamlit + BGE-M3 + RAG")
    st.markdown("Made with ❤️")

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# 主输入区域
user_input = st.chat_input("请在下方输入需要核查的新闻...", disabled=st.session_state.processing)

if user_input and not st.session_state.processing:
    # 设置处理状态
    st.session_state.processing = True
    
    # 将用户消息添加到聊天历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 创建助手消息容器
    with st.chat_message("assistant"):
        # 使用单个容器来避免DOM冲突
        response_container = st.container()
        
        try:
            # 初始化FactChecker
            fact_checker = FactChecker(api_base, model_option, temperature, max_tokens)
            
            # 完整的响应内容
            full_response = ""
            
            # 第1步：提取声明
            with response_container:
                st.markdown('<div class="status-box">🔍 正在提取新闻的核心声明...</div>', unsafe_allow_html=True)
            
            claim = fact_checker.extract_claim(user_input)
            
            # 处理claim字符串
            if "claim:" in claim.lower():
                claim = claim.split("claim:")[-1].strip()
            
            claim_section = f"""
            <div class="evidence-item">
                <h3>🎯 提取新闻的核心声明</h3>
                <p>{claim}</p>
            </div>
            """
            full_response += claim_section
            
            # 第2步：提取关键信息
            with response_container:
                st.markdown(full_response, unsafe_allow_html=True)
                st.markdown('<div class="status-box">🔍 正在提取新闻的关键信息...</div>', unsafe_allow_html=True)
            
            information = fact_checker.extract_keyinformation(user_input)
            
            info_section = f"""
            <div class="evidence-item">
                <h3>🔑 提取新闻的关键信息</h3>
                <p>{information}</p>
            </div>
            """
            full_response += info_section
            
            # 第3步：搜索证据
            with response_container:
                st.markdown(full_response, unsafe_allow_html=True)
                st.markdown('<div class="status-box">🌐 正在搜索相关证据...</div>', unsafe_allow_html=True)
            
            evidence_docs = fact_checker.search_evidence(claim)
            
            # 第4步：获取相关证据块
            with response_container:
                st.markdown(full_response, unsafe_allow_html=True)
                st.markdown('<div class="status-box">🌐 正在分析证据相关性...</div>', unsafe_allow_html=True)
            
            evidence_chunks = fact_checker.get_evidence_chunks(evidence_docs, claim)
            
            # 构建证据部分
            evidence_section = '<div class="evidence-item"><h3>🔗 证据来源</h3>'
            for j, chunk in enumerate(evidence_chunks[:-1]):
                evidence_section += f"""
                <div style="margin: 1rem 0; padding: 0.5rem; background-color: #ffffff; border-radius: 5px;">
                    <strong>[{j+1}]:</strong><br>
                    {chunk['text']}<br>
                    <small><strong>来源:</strong> {chunk['source']}</small>
                </div>
                """
            evidence_section += '</div>'
            full_response += evidence_section
            
            # 第5步：评估声明
            with response_container:
                st.markdown(full_response, unsafe_allow_html=True)
                st.markdown('<div class="status-box">⚖️ 正在评估声明真实性...</div>', unsafe_allow_html=True)
            
            evaluation = fact_checker.evaluate_claim(information, user_input, evidence_chunks)
            
            # 确定结论样式
            verdict = evaluation["verdict"]
            if verdict.upper() == "TRUE":
                emoji = "✅"
                verdict_cn = "正确"
                verdict_class = "verdict-true"
            elif verdict.upper() == "FALSE":
                emoji = "❌"
                verdict_cn = "错误"
                verdict_class = "verdict-false"
            elif verdict.upper() == "PARTIALLY TRUE":
                emoji = "⚠️"
                verdict_cn = "部分正确"
                verdict_class = "verdict-partial"
            else:
                emoji = "❓"
                verdict_cn = "无法验证"
                verdict_class = "verdict-unknown"
            
            # 构建最终结论
            verdict_section = f"""
            <div class="{verdict_class}">
                <h3>{emoji} 结论: {verdict_cn}</h3>
                <h4>📋 推理过程</h4>
                <p>{evaluation['reasoning']}</p>
            </div>
            """
            full_response += verdict_section
            
            # 显示最终完整响应
            with response_container:
                st.markdown(full_response, unsafe_allow_html=True)
            
            # 添加助手响应到聊天历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"""
            <div class="verdict-false">
                <h3>❌ 处理错误</h3>
                <p>抱歉，处理您的请求时出现错误：{str(e)}</p>
                <p>请检查您的配置或稍后重试。</p>
            </div>
            """
            with response_container:
                st.markdown(error_message, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        finally:
            # 重置处理状态
            st.session_state.processing = False
            st.rerun()

# 添加页脚
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        <p>🔍 AI虚假新闻检测器 | 基于大语言模型的新闻真实性验证工具</p>
    </div>
    """, 
    unsafe_allow_html=True
)