import json
from collections import defaultdict
from config import EVAL_DATA_PATH, RECALL_TOP_K
import os  # 新增os模块导入
import streamlit as st  # 新增streamlit导入
from milvus_utils import search_similar_documents  # 新增search_similar_documents导入

def calculate_recall(retrieved_ids, relevant_ids, top_k_list):
    results = {}
    for k in top_k_list:
        top_k_ids = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = top_k_ids & relevant_set
        results[f"recall@{k}"] = len(intersection) / len(relevant_set) if relevant_set else 0
    return results

def load_eval_data(filepath):
    try:
        # 新增路径存在性检查
        if not os.path.exists(filepath):
            st.error(f"评估数据文件不存在：{filepath}")
            return []
            
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"评估数据格式错误：{str(e)}")
        return []
    except Exception as e:
        st.error(f"评估数据加载失败：{str(e)}")
        return []

def run_evaluation(client, embedding_model, top_k_list=RECALL_TOP_K):
    eval_data = load_eval_data(EVAL_DATA_PATH)
    if not eval_data:
        st.error("评估数据集为空，请检查数据文件")
        return {}
    
    metrics = defaultdict(list)
    
    for case in eval_data:
        query = case["query"]
        relevant_ids = case["relevant_ids"]
        
        # 执行检索
        retrieved_ids, _ = search_similar_documents(client, query, embedding_model)
        
        # 新增检索结果验证
        if not retrieved_ids:
            st.warning(f"未检索到相关文档：query='{query}'")
            continue
        
        # 计算指标
        recalls = calculate_recall(retrieved_ids, relevant_ids, top_k_list)
        
        # 记录结果
        for metric, value in recalls.items():
            metrics[metric].append(value)
    
    # 计算平均指标
    final_metrics = {}
    for metric, values in metrics.items():
        final_metrics[metric] = sum(values) / len(values) if values else 0
    
    return final_metrics