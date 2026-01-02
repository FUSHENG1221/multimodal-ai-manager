import chromadb
from chromadb.config import Settings
import os
from config import *


class VectorDB:
    def __init__(self, text_dimension=384, image_dimension=512):
        """初始化向量数据库客户端
        创建或连接到ChromaDB数据库，设置文本和图像的向量集合。
        支持余弦相似度搜索，适用于多模态文档管理。

        Args:
            text_dimension (int, optional): 文本嵌入向量的维度，默认为384
            image_dimension (int, optional): 图像嵌入向量的维度，默认为512

        Attributes:
            client: ChromaDB持久化客户端实例
            text_dimension: 文本向量维度配置
            image_dimension: 图像向量维度配置
            documents_collection: 文档向量集合
            images_collection: 图像向量集合
        """
        # 初始化ChromaDB持久化客户端
        # 设置数据库存储路径和配置参数
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR,  # 数据库存储目录
            settings=Settings(
                anonymized_telemetry=False,  # 禁用匿名遥测数据收集
                persist_directory=VECTOR_DB_DIR  # 数据持久化目录
            )
        )

        # 设置向量维度配置
        self.text_dimension = text_dimension  # 文本嵌入向量维度
        self.image_dimension = image_dimension  # 图像嵌入向量维度

        # 创建或获取文档集合
        # 使用余弦相似度作为距离度量标准
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={
                "hnsw:space": "cosine",  # 使用余弦相似度进行向量搜索
                "description": "学术文档和论文的向量存储",  # 集合描述
                "vector_dimension": text_dimension  # 记录向量维度信息
            }
        )

        # 创建或获取图像集合
        # 同样使用余弦相似度进行图像检索
        self.images_collection = self.client.get_or_create_collection(
            name="images",
            metadata={
                "hnsw:space": "cosine",  # 使用余弦相似度进行向量搜索
                "description": "图像文件的向量存储",  # 集合描述
                "vector_dimension": image_dimension  # 记录向量维度信息
            }
        )

    def add_document(self, document_id, embedding, metadata):
        """添加论文到向量数据库"""
        # 1. 向量维度验证和自动调整
        # 检查嵌入向量的维度是否与配置的文本维度匹配
        if len(embedding) != self.text_dimension:
            # 维度不匹配时的处理策略：
            # - 如果向量过长：截断前text_dimension个维度
            # - 如果向量过短：用0填充到text_dimension维度
            embedding = embedding[:self.text_dimension] + [0] * max(0, self.text_dimension - len(embedding))

        self.documents_collection.add(
            ids=[document_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def add_image(self, image_id, embedding, metadata):
        """添加图像到向量数据库"""
        # 向量维度验证和自动调整
        # 检查嵌入向量的实际维度是否与配置的图像维度匹配
        if len(embedding) != self.image_dimension:
            # 维度不匹配时的智能处理策略：
            # - 向量过长：截取前image_dimension个维度（保留主要特征）
            # - 向量过短：用0填充到目标维度（保持数据结构）
            embedding = embedding[:self.image_dimension] + [0] * max(0, self.image_dimension - len(embedding))

        self.images_collection.add(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def search_documents(self, query_embedding, top_k=TOP_K_RESULTS):
        """搜索相似的论文"""
        # 1. 查询向量维度验证和自动调整
        # 确保查询向量的维度与数据库配置的文本维度一致
        if len(query_embedding) != self.text_dimension:
            # 维度不匹配处理策略：
            # - 截断：如果查询向量维度大于配置维度，截取前text_dimension个元素
            # - 填充：如果查询向量维度小于配置维度，用0填充到目标维度
            query_embedding = query_embedding[:self.text_dimension] + [0] * max(0, self.text_dimension - len(query_embedding))

        # 2. 执行向量相似度搜索
        # 使用ChromaDB的query方法进行相似度查询
        results = self.documents_collection.query(
            query_embeddings=[query_embedding],  # 查询向量列表（支持批量，当前单个）
            n_results=top_k,  # 返回结果数量
            include=["metadatas", "distances"]  # 包含元数据和距离信息
        )

        # 3. 返回标准化格式的搜索结果
        # 结果格式说明：
        # - ids: [[id1, id2, ..., id_top_k]] （二维列表，支持批量查询）
        # - metadatas: [[metadata1, metadata2, ...]] （与ID对应的元数据）
        # - distances: [[dist1, dist2, ...]] （余弦距离，值越小越相似）
        return results

    def search_images(self, query_embedding, top_k=5):
        """搜索相似的图像"""
        # 1. 查询向量维度验证和自动调整
        # 确保查询向量的维度与数据库配置的图像维度一致
        if len(query_embedding) != self.image_dimension:
            # 维度不匹配处理策略：
            # - 截断：如果查询向量维度大于配置维度，截取前image_dimension个元素
            # - 填充：如果查询向量维度小于配置维度，用0填充到目标维度
            query_embedding = query_embedding[:self.image_dimension] + [0] * max(0, self.image_dimension - len(query_embedding))

        # 2. 执行向量相似度搜索
        # 使用ChromaDB的query方法在图像集合中进行相似度查询
        results = self.images_collection.query(
            query_embeddings=[query_embedding],  # 查询向量列表（支持批量查询）
            n_results=top_k,  # 返回最相似的top_k个结果
            include=["metadatas", "distances"]  # 包含元数据和距离信息
        )

        # 3. 返回标准化格式的搜索结果
        # 结果格式说明：
        # - ids: [[id1, id2, ..., id_top_k]] （二维列表，支持批量查询）
        # - metadatas: [[metadata1, metadata2, ...]] （与ID对应的元数据）
        # - distances: [[dist1, dist2, ...]] （余弦距离，0=完全相同，2=完全相反）
        return results

    def get_all_documents(self):
        """获取所有论文"""
        results = self.documents_collection.get(include=["metadatas"])
        return results

    def get_all_images(self):
        """获取所有图像"""
        results = self.images_collection.get(include=["metadatas"])
        return results

    def delete_document(self, document_id):
        """删除论文"""
        self.documents_collection.delete(ids=[document_id])

    def delete_image(self, image_id):
        """删除图像"""
        self.images_collection.delete(ids=[image_id])

    def clear_documents(self):
        """清空所有论文数据"""
        all_documents = self.get_all_documents()
        if 'ids' in all_documents and all_documents['ids']:
            document_ids = all_documents['ids']
            self.documents_collection.delete(ids=document_ids)

    def clear_images(self):
        """清空所有图像数据"""
        all_images = self.get_all_images()
        if 'ids' in all_images and all_images['ids']:
            image_ids = all_images['ids']
            self.images_collection.delete(ids=image_ids)