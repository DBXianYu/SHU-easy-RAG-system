import os
import umap
import plotly.express as px
import numpy as np
from sklearn.cluster import DBSCAN  # 新增：导入DBSCAN聚类算法
from milvus_utils import get_milvus_client
from config import MILVUS_LITE_DATA_PATH, COLLECTION_NAME

def visualize_milvus_vectors():
    # 1. 连接Milvus
    client = get_milvus_client()
    if not client:
        print("无法连接Milvus客户端")
        return

    # 2. 查询所有向量数据（示例：获取前2000条，避免数据量过大）
    print("正在从Milvus查询向量数据...")
    res = client.query(
        collection_name=COLLECTION_NAME,
        filter="id < 10000",  # 限制查询数量，可根据需要调整
        output_fields=["embedding", "content_preview"]  # 获取向量和元数据
    )

    # 提取向量和元数据
    vectors = np.array([item["embedding"] for item in res])
    labels = [item["content_preview"][:50] for item in res]  # 使用content_preview作为标签

    if len(vectors) == 0:
        print("Milvus集合中无数据")
        return

    # 3. 降维（UMAP）
    print("正在降维处理...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(vectors)  # 结果为3D坐标

    # 4. 基于原始高维向量进行DBSCAN聚类（按距离聚类）
    print("正在聚类...")
    # 调整eps参数（向量间的最大距离阈值）和min_samples（核心点所需最小样本数）
    dbscan = DBSCAN(eps=0.2, min_samples=3, leaf_size = 300, metric="euclidean")  # leaf_size为
    cluster_labels = dbscan.fit_predict(vectors)  # 对原始高维向量聚类

    # 5. 可视化（按聚类结果着色）
    print("生成可视化图表...")
    fig = px.scatter_3d(
        x=embedding_3d[:, 0],
        y=embedding_3d[:, 1],
        z=embedding_3d[:, 2],
        hover_data={"标签": labels, "聚类标签": cluster_labels},  # 悬停显示聚类标签
        title="Milvus向量库3D可视化（按距离聚类着色）",
        labels={"x": "UMAP 维度1", "y": "UMAP 维度2", "z": "UMAP 维度3"},
        color=cluster_labels.astype(str),  # 转换为字符串避免颜色梯度连续
        color_discrete_sequence=px.colors.qualitative.Plotly  # 使用离散颜色序列
    )
    fig.update_traces(marker=dict(size=1))
    fig.show()

if __name__ == "__main__":
    visualize_milvus_vectors()