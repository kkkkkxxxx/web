import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
import dashscope  # 替换 OpenAI
from dashscope import Generation  # 替换 OpenAI
import requests
from duckduckgo_search import DDGS
import numpy as np
import re
from FlagEmbedding import BGEM3FlagModel

class FactChecker:
    
    def __init__(self, api_base: str, model: str, temperature: float, max_tokens: int):
        """
        初始化事实核查器，设置配置参数
        
        参数:
            api_base: LLM API的基础URL（未使用，已弃用）
            model: 用于事实核查的模型（需替换为 DashScope 支持的模型名称）
            temperature: LLM的温度参数
            max_tokens: LLM响应的最大token数
        """
        # ✅ 设置 DashScope API Key（需替换为你的阿里云 API Key）
        self.dashscope_api_key = "sk-8f7cf717823e4a718c0b434b74947478"  # 替换为你的阿里云 API Key
        dashscope.api_key = self.dashscope_api_key
        
        self.model = model  # 使用 DashScope 支持的模型名称（如 qwen2-7b-instruct-ft-202505022309-1958）
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化嵌入模型
        try:
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        except Exception as e:
            st.error(f"加载BGE-M3模型错误: {str(e)}")
            self.embedding_model = None
            
        # 初始化Chroma本地知识库
        try:
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="my-knowledge-base", 
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
            )
        except Exception as e:
            st.error(f"加载Chroma错误: {str(e)}")
            self.collection = None
            
    def extract_claim(self, text: str) -> str:
        """
        Extract core claims from the input text using LLM.
        
        Args:
            text: The input text to extract claims from
            
        Returns:
            extracted claim
        """
        system_prompt = """
        You are a precise claim extraction assistant. Analyze the provided news and summarize the central idea of it.
        Format the central idea as a worthy-check statement, which is a claim that can be verified independently.
        output format:
        claim: <claim>
        """
        #You are a precise claim extraction assistant. Analyze the provided news and summarize the central idea of it.
        #Format the central idea as a worthy-check statement, which is a claim that can be verified independently.
        #output format:
        #claim: <claim>
        try:
            # ✅ 使用 DashScope 的 Generation.call 替代 OpenAI
            prompt = f"{system_prompt}\n\n用户输入: {text}"
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=1000
            )
            
            claims_text = response.output.text
            
            # 将编号列表解析为单独的声明
            claims = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', claims_text, re.DOTALL)
            
            # 清理声明
            claims = [claim.strip() for claim in claims if claim.strip()]
            
            # 如果没有找到编号声明，则按换行符分割
            if not claims and claims_text.strip():
                claims = [line.strip() for line in claims_text.strip().split('\n') if line.strip()]
            
            return claims[0] if claims else text
            
        except Exception as e:
            st.error(f"提取声明错误: {str(e)}")
            return text  # 作为回退返回原始文本

    def extract_keyinformation(self, text: str) -> str:
        """
        使用LLM从输入文本中提取核心声明及其上下文信息
    
        参数:
            text: 要从中提取声明的输入文本
        
        返回:
            提取的结构化信息字符串
        """
        system_prompt = """
        You are a precise claim extraction assistant. Please perform the following multi-level analysis on the user input:
    
        1. Identify all entities involved in the text (person/organization/location/time) and their types
        2. Determine the primary event category (political/economic/social/technological etc.)
        3. Extract key characteristics (source reliability/evidence strength/timeliness)
        4. Summarize the central idea and convert it into a verifiable claim
    
        Output format requirements:
        claim: <Verifiable complete statement>
        entities: <Entity type1:Entity name1, Entity type2:Entity name2...>
        event_type: <Primary event category>
        key_features: <Feature1:Value1, Feature2:Value2...>
    
        Note: Please strictly follow the format and avoid additional explanations
       """
    
        try:
        # ✅ 使用 DashScope 的 Generation.call 替代 OpenAI
          prompt = f"{system_prompt}\n\n用户输入: {text}"
          response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1000
         )
        
          raw_output = response.output.text
        
        # 使用正则表达式精确提取各字段
          claim_match = re.search(r'claim:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
          entities_match = re.search(r'entities:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
          event_type_match = re.search(r'event_type:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
          key_features_match = re.search(r'key_features:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
        
        # 清理提取的字段内容
          claim = claim_match.group(1).strip() if claim_match else "未提取到有效声明"
          entities = entities_match.group(1).strip() if entities_match else "无实体信息"
          event_type = event_type_match.group(1).strip() if event_type_match else "未知事件类型"
          key_features = key_features_match.group(1).strip() if key_features_match else "无特征信息"
        
        # 格式化输出
          formatted_output = (
            f"claim: {claim}\n"
            f"entities: {entities}\n"
            f"event_type: {event_type}\n"
            f"key_features: {key_features}"
          )
        
          return formatted_output
            
        except Exception as e:
          st.error(f"提取声明错误: {str(e)}")
        return text  # 作为回退返回原始文本
    
    def search_local_knowledge(self, claim: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Search for evidence from local Chroma knowledge base.

        Args:
            claim: The claim to search for
            top_k: Number of top results to return

        Returns:
               List of evidence documents
        """
        if not self.collection:
            return []
    
        try:
            results = self.collection.query(
                query_texts=[claim],
                n_results=top_k
            )
            evidence_docs = []
            for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
                evidence_docs.append({
                   'title': metadata.get('title', ''),
                    'url': metadata.get('url', 'Local Knowledge Base'),
                    'snippet': text
            })
            return evidence_docs
        except Exception as e:
            st.error(f"Error searching local knowledge base: {str(e)}")
            return []
        
    def search_evidence(self, claim: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for evidence using DuckDuckGo.
        
        Args:
            claim: The claim to search for evidence
            num_results: Number of search results to return
            
        Returns:
            List of evidence documents with title, url, and snippet
        """
        try:
            ddgs = DDGS(timeout=60)#proxy="socks5://127.0.0.1:20170",
            results = list(ddgs.text(claim, max_results=num_results))
            
            external_evidence = []#此处将evidence_docs改为external_evidence
            for result in results:
                #此处将evidence_docs改为external_evidence
                external_evidence.append({      
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            # 2. 本地知识库检索
            local_evidence = self.search_local_knowledge(claim, top_k=num_results)
            # 3. 合并两种来源
            combined_evidence = external_evidence + local_evidence
            return combined_evidence
            #return evidence_docs
        except Exception as e:
            st.error(f"Error searching for evidence: {str(e)}")
            return []

    #下面evidence_docs改为external_evidence
    def get_evidence_chunks(self,evidence_docs: List[Dict[str, str]], claim: str, chunk_size: int = 200, chunk_overlap: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract and rank evidence chunks related to the claim using BGE-M3.
        
        Args:
            evidence_docs: List of evidence documents
            claim: The claim to match with evidence
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of top chunks to return
            
        Returns:
            List of ranked evidence chunks with similarity scores
        """
        if not self.embedding_model:
            return [{
                'text': "Evidence ranking unavailable - BGE-M3 model could not be loaded.",
                'source': "System",
                'similarity': 0.0
            }]
        
        #检查 evidence_docs 是否为空在处理证据片段之前，确保 evidence_docs 列表不是空的。
        #如果在搜索过程中没有找到证据，需要优雅地处理这种情况，可以返回一个默认消息或跳过片段处理。
        if not evidence_docs:  # 检查是否没有找到证据
            return [{
            'text': "没有找到相关证据。",
            'source': "系统",
            'similarity': 0.0
            }]
        
        try:
            # Create text chunks from evidence documents
            all_chunks = []
            
            for doc in evidence_docs:
                # Add title as a separate chunk
                all_chunks.append({
                    'text': doc['title'],
                    'source': doc['url'],
                })
                
                # Process the snippet into overlapping chunks
                snippet = doc['snippet']
                if len(snippet) <= chunk_size:
                    # If snippet is shorter than chunk_size, use it as is
                    all_chunks.append({
                        'text': snippet,
                        'source': doc['url'],
                    })
                else:
                    # Create overlapping chunks
                    for i in range(0, len(snippet), chunk_size - chunk_overlap):
                        chunk_text = snippet[i:i + chunk_size]
                        if len(chunk_text) >= chunk_size // 2:  # Only keep chunks of reasonable size
                            all_chunks.append({
                                'text': chunk_text,
                                'source': doc['url'],
                            })
            
            # Compute embeddings for claim
            claim_embedding = self.embedding_model.encode(claim)['dense_vecs']
            
            # Compute embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)['dense_vecs']
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(claim_embedding, chunk_embedding) / (
                    np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(float(similarity))
            
            # Add similarities to chunks
            for i, similarity in enumerate(similarities):
                all_chunks[i]['similarity'] = similarity
            
            # Sort chunks by similarity (descending)
            ranked_chunks = sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)
            
            # Return top k chunks
            return ranked_chunks[:top_k]
            
        except Exception as e:
            st.error(f"Error ranking evidence: {str(e)}")
            return [{
                'text': f"Error ranking evidence: {str(e)}",
                'source': "System",
                'similarity': 0.0
            }]
    def evaluate_claim(self, keyinformation: str, claim: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Evaluate the truthfulness of a claim based on evidence using LLM.
        
        Args:
            claim: The claim to evaluate
            evidence_chunks: The evidence chunks to use for evaluation
            
        Returns:
            Dictionary with verdict and reasoning
        """
        system_prompt = """
        [Role] Fake News Verifier  
[Task] Strictly follow this analysis protocol:  

1. **Source Credibility Verification (Weight: 40%)**  
   ✓ Official sources (government/academic/professional journals) → 5★  
   ✓ Mainstream media (AFP/Xinhua, etc.) → 3★  
   ✗ Personal blogs/anonymous platforms → 1★  

2. **Evidence Quality Assessment (Weight: 40%)**  
   │ Direct Evidence      │ Indirect Evidence  
   ├─────────────────────┼─────────────────────┤  
   │ Experimental data   │ Expert testimony  
   │ Statistical reports │ Historical cases  
   │ On-site footage     │ Literature citations  

3. **Logical Completeness Review (Weight: 20%)**  
   - Statistical trap detection: Sample size <30  
   - Causality fallacy identification: A→B misrepresented as B→A  
   - Temporal paradox verification: Event interval <2h  

4. **Binary Judgment Criteria**  
   **TRUE**: Total weight ≥70% with no core contradictions  
   **FALSE**: Existence of counter-evidence chains/multiple logical flaws or majority sourcing from personal blogs/anonymous platforms  

[Output Specification]  
**VERDICT**: [TRUE|FALSE|PARTIALLY TRUE]  
**EVIDENCE WEIGHT**: [Total evidence weight percentage]  
**REASONING**:  
1. Source rating: □Official □Verified media □Suspicious platform  
2. Key evidence: [E-12] NASA remote sensing data (2024Q3)  
3. Logical flaw: Sample selection bias detected (p=0.12)   
4. Temporal validity: ✓ Latest satellite imagery (2025-03)  

**Input**: "There's an alien base on the far side of the Moon"  
**Output**:  
VERDICT: FALSE  
EVIDENCE WEIGHT: 92%  
REASONING:  
1. Source rating: NASA official data (Weight: 50%)  
2. Key evidence: [E-3] Lunar rover scans show no anomalous structures  
3. Logical flaw: Eyewitness accounts constitute anecdotal fallacy  
4. Temporal verification: Latest lunar mission data (2024Q3)  
        """
    
        # Prepare evidence text for the prompt
        evidence_text = "\n\n".join([
            f"evidence {i+1} (relevance: {chunk['similarity']:.2f}):\n{chunk['text']}\nsource: {chunk['source']}"
            for i, chunk in enumerate(evidence_chunks)
        ])

        try:
            prompt = f"{system_prompt}\n\nkeyinformation: {keyinformation}\n\ntext:{claim}\n\nevidence:\n{evidence_text}"
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.output.text
            # 清理特殊空白字符（处理全角空格/零宽空格/换行）
            result_text_clean = re.sub(r'[\u3000\u200B\xa0]+', ' ', result_text)
            result_text_clean = re.sub(r'\s+', ' ', result_text_clean)  # 统一空白格式
            
            verdict_match = re.search(r'\s*(TRUE|FALSE|PARTIALLY TRUE).*?$', result_text, re.IGNORECASE | re.MULTILINE)
            verdict = verdict_match.group(1) if verdict_match else "UNVERIFIABLE"
            
            reasoning_match = re.search(r'REASONING:\s*(.*)', result_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text
            
            return {
                "verdict": verdict,
                "reasoning": reasoning
            }
            
        except Exception as e:
            st.error(f"评估声明错误: {str(e)}")
            return {
                "verdict": "错误",
                "reasoning": f"评估过程中发生错误: {str(e)}"
            }
            
        except Exception as e:
            st.error(f"Error evaluating claim: {str(e)}")
            return {
                "verdict": "ERROR",
                "reasoning": f"An error occurred during evaluation: {str(e)}"
            }

    def check_fact(self, text: str) -> Dict[str, Any]:
        """
        Main function to check the factuality of a statement.
        
        Args:
            text: The statement to fact-check
            
        Returns:
            Dictionary with all results of the fact-checking process
        """
        # 1. Extract core claim
        claim = self.extract_claim(text)
        
        result = {
            "original_text": text,
            "claim": claim,
            "results": []
        }
        
        # 搜集关键信息
        key=self.extract_keyinformation(text)
        
        # 2. Search for evidence
        evidence_docs = self.search_evidence(claim)
        
        # 3. Get relevant evidence chunks
        evidence_chunks = self.get_evidence_chunks(evidence_docs, claim)
        
        # 4. Evaluate claim based on evidence
        evaluation = self.evaluate_claim(key, claim, evidence_chunks)
        
        # Add results for this claim
        result={
            "claim": claim,
            "evidence_docs": evidence_docs,
            "evidence_chunks": evidence_chunks,
            "verdict": evaluation["verdict"],
            "reasoning": evaluation["reasoning"]
        }
        
        return result


# Function to be imported in the main Streamlit app
def check_fact(claim: str, api_base: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Public interface for fact checking to be used by the Streamlit app.
    
    Args:
        claim: The statement to fact-check
        api_base: The base URL for the LLM API
        model: The model to use for fact checking
        temperature: Temperature parameter for LLM
        max_tokens: Maximum tokens for LLM response
        
    Returns:
        Dictionary with all results of the fact-checking process
    """
    checker = FactChecker(api_base, model, temperature, max_tokens)
    return checker.check_fact(claim)