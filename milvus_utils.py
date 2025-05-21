import streamlit as st
# Use MilvusClient for Lite version
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os

# Import config variables including the global map
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    SEARCH_PARAMS, TOP_K, id_to_doc_map
)

@st.cache_resource
def get_milvus_client():
    """Initializes and returns a MilvusClient instance for Milvus Lite."""
    try:
        st.write(f"Initializing Milvus Lite client with data path: {MILVUS_LITE_DATA_PATH}")
        # Ensure the directory for the data file exists
        os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH), exist_ok=True)
        # The client connects to the local file specified
        client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
        st.success("Milvus Lite client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Milvus Lite client: {e}")
        return None

import pymilvus 

from pymilvus import CollectionSchema, FieldSchema, DataType  # 确保已导入

# @st.cache_resource(
#     hash_funcs={
#         pymilvus.milvus_client.milvus_client.MilvusClient: lambda _: None
#     },
#     show_spinner=False
# )

# def setup_milvus_collection(client):
#     """Creates or loads the Milvus collection with proper schema."""
#     collection_name = COLLECTION_NAME
#     try:
#         # 使用 FieldSchema 和 CollectionSchema 构建标准 Schema
#         fields = [
#             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
#             FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),  # dim 明确为整数
#             FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=2000),
#             FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256)
#         ]
#         schema = CollectionSchema(fields=fields, auto_id=False)  # 正确的 Schema 对象

#         if client.has_collection(collection_name):
#             st.write(f"Collection '{collection_name}' already exists.")
#             # indexes = client.list_indexes(collection_name)

#             # from pymilvus.milvus_client.index import IndexParams
#             # index_params = IndexParams(
#             #     index_type=INDEX_TYPE,
#             #     metric_type=INDEX_METRIC_TYPE,
#             #     params=INDEX_PARAMS  # 直接传递参数字典
#             # )
#             # embedding_indexes = [idx for idx in indexes if idx["field_name"] == "embedding"]
#             # if not embedding_indexes:
#             #     st.write(f"Index for 'embedding' field not found. Creating index...")
#             #     client.create_index(
#             #         collection_name=collection_name,
#             #         field_name="embedding",
#             #         index_type=INDEX_TYPE,
#             #         metric_type=INDEX_METRIC_TYPE,
#             #         index_params=index_params  # 直接传递参数字典
#             #     )
#             #     st.success("Index created successfully.")
#             # else:
#             #     st.write(f"Index for 'embedding' field already exists: {embedding_indexes[0]}")
#             # return True
#             return True
#         else:
#             st.write(f"Creating collection '{collection_name}'...")
#             client.create_collection(collection_name, schema=schema)  # 传递 Schema 对象
#             # 修正索引创建参数传递方式（明确传递各参数）
#             # client.create_index(collection_name, "embedding", {"index_type": INDEX_TYPE, "metric_type": INDEX_METRIC_TYPE, "params": INDEX_PARAMS})
#             return True

            
#     except Exception as e:
#         st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
#         return False

@st.cache_resource
def setup_milvus_collection(_client):
    """Creates or loads the Milvus collection with proper schema."""
    collection_name = COLLECTION_NAME
    try:
        # 使用Milvus官方Schema类定义数据结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256)
        ]
        schema = CollectionSchema(fields=fields)

        if _client.has_collection(collection_name):
            st.write(f"Collection '{collection_name}' already exists.")
            # 修复：检查 list_indexes 返回值是否为有效列表
            indexes = _client.list_indexes(collection_name) or []
            # 修复：确保索引项是字典且包含 'field_name' 键（或根据实际API调整键名）
            embedding_indexes = [
                idx for idx in indexes 
                if isinstance(idx, dict) and idx.get("field_name") == "embedding"
            ]
            if not embedding_indexes:
                st.write(f"Index for 'embedding' field not found. Creating index...")
                from pymilvus.milvus_client.index import IndexParam, IndexParams  # 导入正确的类
                # 使用 IndexParam 定义单个索引参数（注意是单数）
                index_param = IndexParam(
                    field_name="embedding",
                    index_type=INDEX_TYPE,
                    index_name="embedding_index",  # 可选：指定索引名
                    metric_type=INDEX_METRIC_TYPE,
                    **INDEX_PARAMS  # 展开具体参数（如 nlist=128）
                )
                index_params = IndexParams([index_param])  # IndexParams 是存储 IndexParam 的列表

                # 修复：移除显式的 field_name 参数，避免重复传递
                _client.create_index(
                    collection_name=collection_name,
                    index_params=index_params  # 仅传递 index_params
                )
                st.success("Index created successfully.")
            _client.load_collection(collection_name)
            return True
        else:
            st.write(f"Creating collection '{collection_name}'...")
            _client.create_collection(collection_name, schema=schema)
            
            # 修复索引参数创建逻辑
            from pymilvus.milvus_client.index import IndexParam, IndexParams
            index_param = IndexParam(
                field_name="embedding",
                index_type=INDEX_TYPE,
                index_name="embedding_index",
                metric_type=INDEX_METRIC_TYPE,
                **INDEX_PARAMS
            )
            index_params = IndexParams([index_param])
            
            # 修复：移除显式的 field_name 参数，避免重复传递
            _client.create_index(
                collection_name=collection_name,
                index_params=index_params  # 仅传递 index_params
            )
            _client.load_collection(collection_name)
            return True
    except Exception as e:
        st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
        return False



# def index_data_if_needed(client, data, embedding_model):
#     """Checks if data needs indexing and performs it using MilvusClient."""
#     global id_to_doc_map # Modify the global map

#     if not client:
#         st.error("Milvus client not available for indexing.")
#         return False

#     collection_name = COLLECTION_NAME
#     if not setup_milvus_collection(client):
#         return False
#     # Retrieve current entity count with fallback

    

#     try:
#         if hasattr(client, 'num_entities'):
#             current_count = client.num_entities(collection_name)
#         else:
#             stats = client.get_collection_stats(collection_name)
#             current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
#     except Exception:
#         st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
#         if not setup_milvus_collection(client):
#             return False
#         current_count = 0  # Assume empty after setup

#     st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

#     data_to_index = data[:MAX_ARTICLES_TO_INDEX] # Limit data for demo
#     needed_count = 0
#     docs_for_embedding = []
#     data_to_insert = [] # List of dictionaries for MilvusClient insert
#     temp_id_map = {} # Build a temporary map first

#     # Prepare data
#     with st.spinner("Preparing data for indexing..."):
#         for i, doc in enumerate(data_to_index):
#             title = doc.get('title', '') or ""
#             abstract = doc.get('abstract', '') or ""
#             content = f"Title: {title}\nAbstract: {abstract}".strip()
#             if not content:
#                 continue

#             doc_id = i # Use list index as ID
#             needed_count += 1
#             temp_id_map[doc_id] = {
#                 'title': title, 
#                 'abstract': abstract, 
#                 'content': content,
#                 'source_file': doc.get('source_file')  # 从 doc 中获取 source_file（确保 doc 包含该字段）
#             }
#             docs_for_embedding.append(content)
#             # Prepare data in dict format for MilvusClient（新增 source_file 字段）
#             data_to_insert.append({
#                 "id": doc_id,
#                 "embedding": None, # Placeholder, will be filled after encoding
#                 "content_preview": content[:2000],
#                 "source_file": doc.get('source_file')  # 新增：从 doc 中获取 source_file
#             })


#     if current_count < needed_count and docs_for_embedding:
#         st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

#         st.write(f"Embedding {len(docs_for_embedding)} documents...")
#         with st.spinner("Generating embeddings..."):
#             start_embed = time.time()
#             embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
#             end_embed = time.time()
#             st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

#         # Fill in the embeddings
#         for i, emb in enumerate(embeddings):
#             data_to_insert[i]["embedding"] = emb

#         st.write("Inserting data into Milvus Lite...")
#         with st.spinner("Inserting..."):
#             try:
#                 start_insert = time.time()
#                 # MilvusClient uses insert() with list of dicts
#                 res = client.insert(collection_name=collection_name, data=data_to_insert)
#                 # Milvus Lite might automatically flush or sync, explicit flush isn't usually needed/available
#                 end_insert = time.time()
#                 # 使用 len(data_to_insert) 作为成功插入的数量，因为 res 可能没有 primary_keys 属性
#                 inserted_count = len(data_to_insert)
#                 st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
#                 # Update the global map ONLY after successful insertion attempt
#                 id_to_doc_map.update(temp_id_map)
#                 return True
#             except Exception as e:
#                 st.error(f"Error inserting data into Milvus Lite: {e}")
#                 return False
#     elif current_count >= needed_count:
#         st.write("Data count suggests indexing is complete.")
#         # Populate the global map if it's empty but indexing isn't needed
#         if not id_to_doc_map:
#             id_to_doc_map.update(temp_id_map)
#         return True
#     else: # No docs_for_embedding found
#          st.error("No valid text content found in the data to index.")
#          return False

def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using MilvusClient."""
    global id_to_doc_map # Modify the global map

    if not client:
        st.error("Milvus client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    # Retrieve current entity count with fallback
    try:
        if hasattr(client, 'num_entities'):
            current_count = client.num_entities(collection_name)
        else:
            stats = client.get_collection_stats(collection_name)
            current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
    except Exception:
        st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
        if not setup_milvus_collection(client):
            return False
        current_count = 0  # Assume empty after setup

    st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX] # Limit data for demo
    needed_count = 0
    docs_for_embedding = []
    data_to_insert = [] # List of dictionaries for MilvusClient insert
    temp_id_map = {} # Build a temporary map first

    # Prepare data
    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
             title = doc.get('title', '') or ""
             abstract = doc.get('abstract', '') or ""
             content = f"Title: {title}\nAbstract: {abstract}".strip()
             if not content:
                 continue

             doc_id = i # Use list index as ID
             needed_count += 1
             temp_id_map[doc_id] = {
                 'title': title, 'abstract': abstract, 'content': content
             }
             docs_for_embedding.append(content)
             # Prepare data in dict format for MilvusClient
             data_to_insert.append({
                 "id": doc_id,
                 "embedding": None, # Placeholder, will be filled after encoding
                 "content_preview": content[:2000], # Store preview if field exists
                 "source_file": doc.get('source_file')  # 新增：从 doc 中获取 source_file
             })


    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        # Fill in the embeddings
        for i, emb in enumerate(embeddings):
            data_to_insert[i]["embedding"] = emb

        st.write("Inserting data into Milvus Lite...")
        # st.write(f"Data to insert: {data_to_insert}")  # 打印插入的数据
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                # MilvusClient uses insert() with list of dicts
                res = client.insert(collection_name=collection_name, data=data_to_insert)
                # Milvus Lite might automatically flush or sync, explicit flush isn't usually needed/available
                end_insert = time.time()
                # 使用 len(data_to_insert) 作为成功插入的数量，因为 res 可能没有 primary_keys 属性
                inserted_count = len(data_to_insert)
                st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                # Update the global map ONLY after successful insertion attempt
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"Error inserting data into Milvus Lite: {e}")
                return False
    elif current_count >= needed_count:
        st.write(f"Data count suggests indexing is complete.{current_count}/{needed_count}")
        # Populate the global map if it's empty but indexing isn't needed
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else: # No docs_for_embedding found
         st.error("No valid text content found in the data to index.")
         return False


def search_similar_documents(client, query, embedding_model):
    """Searches Milvus Lite for documents similar to the query using MilvusClient."""
    if not client or not embedding_model:
        st.error("Milvus client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        query_embedding = embedding_model.encode([query])[0]

        # 重写search调用，使用更兼容的方式
        search_params = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"]
        }
        
        
        # 尝试不同的方式传递搜索参数
        if hasattr(client, 'search_with_params'):
            st.write("存在专门方法")
            # 如果存在专门的方法
            res = client.search_with_params(**search_params, search_params=SEARCH_PARAMS)

        else:
            # 标准方法，直接设置参数（当前版本会导致参数冲突）
            try:
                # 尝试1：不传递param参数
                # res = client.search(**search_params)
                res = client.search(collection_name= collection_name,data= [query_embedding],anns_field= "embedding",limit= TOP_K,output_fields= ["id"])
            except Exception as e1:
                st.warning(f"搜索尝试1失败: {e1}，将尝试备用方法...")
                try:
                    # 尝试2：通过搜索参数关键字传递
                    res = client.search(**search_params, **SEARCH_PARAMS)
                except Exception as e2:
                    st.warning(f"搜索尝试2失败: {e2}，将尝试最后一种方法...")
                    # 尝试3：结合参数
                    final_params = search_params.copy()
                    final_params["nprobe"] = SEARCH_PARAMS.get("nprobe", 16)
                    res = client.search(**final_params)

        # Process results (structure might differ slightly)
        # client.search returns a list of lists of hits (one list per query vector)
        if not res or not res[0]:
            return [], []

        hit_ids = [hit['id'] for hit in res[0]]
        distances = [hit['distance'] for hit in res[0]]
        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during Milvus Lite search: {e}")
        return [], []