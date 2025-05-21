import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY, RERANKING_NUM

def generate_answer(query, context_docs, gen_model, tokenizer, chat_history):  # 修改：新增 chat_history 参数
    """Generates an answer using the LLM based on query, context, and chat history."""
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    chat_history_str = "\n".join([
        f"用户: {msg['content']}" if msg['role'] == 'user' 
        else f"助理: {msg['content']}" 
        for msg in chat_history
    ])


    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs

#If the answer is not found in the context, state that clearly. Do not make up information.
    prompt = f"""Based ONLY on the following context documents, answer the user's question in details, in Chinese. 
If the answer is not found in the context, state that clearly

Chat History:
{chat_history_str}

Context Documents:
{context}

User Question: {query}

Answer:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 




# # def generate_answer(query, context_docs, gen_model, tokenizer, chat_history):  # 修改：新增 chat_history 参数
# def generate_answer(query, docs, model, tokenizer, chat_history, num_candidates=3, temperature=0.7):
#     # 1. 构建上下文提示
#     context = "\n".join([f"文档{i+1}: {doc['content']}" for i, doc in enumerate(docs)])

#     if not docs:
#         return "I couldn't find relevant documents to answer your question."
#     if not model or not tokenizer:
#          st.error("Generation model or tokenizer not available.")
#          return "Error: Generation components not loaded."

#     chat_history_str = "\n".join([
#         f"用户: {msg['content']}" if msg['role'] == 'user' 
#         else f"助理: {msg['content']}" 
#         for msg in chat_history
#     ])


#     context = "\n\n---\n\n".join([doc['content'] for doc in docs]) # Combine retrieved docs

#     final_prompt = f"""Based ONLY on the following context documents, answer the user's question in details, in Chinese. 
# If the answer is not found in the context, state that clearly

# Chat History:
# {chat_history_str}

# Context Documents:
# {context}

# User Question: {query}

# Answer:
# """
    
#     try:
#         # 2. 生成多个候选答案（通过调整temperature增加多样性）
#         inputs = tokenizer(final_prompt, return_tensors="pt")
        
#         inputs = inputs.to(model.device)
        
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS_GEN,
#             num_return_sequences=num_candidates,  # 生成N个候选
#             temperature=temperature,             # 控制随机性
#             top_k=50,                             # 保留top50词
#             top_p=0.95,                           # 核采样阈值
#             do_sample=True                        # 启用采样
#         )
        
#         # 3. 解码并清洗候选答案
#         # candidates = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
#         # candidates = [tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)   for output in outputs]
#         # response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#         candidates = []
#         input_length = inputs['input_ids'].shape[1]  # Sequence length of input prompt
#         for output in outputs:
#             if len(output) <= input_length:  # Handle cases where model didn't generate new tokens
#                 candidates.append("(No new content generated)")
#             else:
#                 decoded = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
#                 candidates.append(decoded)

#         # 4. 对候选答案重排序（使用AI模型评分）
#         if RERANKING_ENABLED and reranking_model and reranking_tokenizer:  # 假设已加载评分模型和tokenizer
#             with torch.no_grad():
#                 # 准备查询-候选对（格式：[(query, candidate), ...]）
#                 sentence_pairs = [(query, candidate) for candidate in candidates]
#                 # 编码输入
#                 inputs = reranking_tokenizer(
#                     sentence_pairs, 
#                     padding=True, 
#                     truncation=True, 
#                     return_tensors="pt",
#                     max_length=512
#                 ).to(reranking_model.device)
#                 # 预测相关性分数（分数越高越相关）
#                 outputs = reranking_model(**inputs)
#                 scores = outputs.logits.softmax(dim=1)[:, 1].tolist()  # 取正类概率
#                 # 按分数降序排序
#                 ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#         else:
#             # 回退到原字符重叠率评分（保留兼容）
#             ranked_candidates = []
#             for candidate in candidates:
#                 if len(set(query)) == 0:
#                     overlap = 0.0
#                 else:
#                     overlap = len(set(query) & set(candidate)) / len(set(query))
#                 ranked_candidates.append((candidate, overlap))
#             ranked_candidates.sort(key=lambda x: x[1], reverse=True)
#         return [item[0] for item in ranked_candidates]  # 返回排序后的候选列表

#     except Exception as e:
#         st.error(f"Error during text generation: {e}")
#         return "Sorry, I encountered an error while generating the answer."


def no_doc_generate_answer(query, gen_model, tokenizer, chat_history):  # 修改：新增 chat_history 参数
    """Generates an answer using the LLM based on query, context, and chat history."""
    # if not context_docs:
    #     return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    chat_history_str = "\n".join([
        f"用户: {msg['content']}" if msg['role'] == 'user' 
        else f"助理: {msg['content']}" 
        for msg in chat_history
    ])


    # context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs

#If the answer is not found in the context, state that clearly. Do not make up information.
    prompt = f"""Answer the user's question in details, in Chinese. 

Chat History:
{chat_history_str}



User Question: {query}

Answer:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 