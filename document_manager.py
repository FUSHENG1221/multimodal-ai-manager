import os
import shutil
import time
import uuid
import PyPDF2
from PIL import Image
from config import *
from ai_client import AIClient
from vector_db import VectorDB


class DocumentManager:
    def __init__(self, text_model_type=DEFAULT_TEXT_MODEL):
        self.text_model_type = text_model_type
        self.ai_client = AIClient(text_model_type=text_model_type)
        self.vector_db = VectorDB(text_dimension=self.ai_client.get_text_embedding_dimension())

    def extract_text_from_pdf(self, pdf_path):
        """从PDF文件中提取文本"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:# 遍历每一页，读取文本内容
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_relevant_fragment(self, pdf_path, query, fragment_length=200):
        """从PDF中提取与查询相关的文本片段"""
        try:
            # 1. 打开并读取PDF文件
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # 2. 初始化最佳匹配结果变量
                best_match_score = 0  # 最高匹配分数
                best_fragment = ""  # 最佳匹配片段
                best_page_num = 1  # 最佳匹配页码

                # 3. 遍历PDF的每一页
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()  # 提取当前页的文本内容

                    # 4. 将页面文本分割为段落（以双换行符为分隔符）
                    paragraphs = text.split('\n\n') if '\n\n' in text else [text]

                    # 5. 遍历每个段落，计算与查询的匹配度
                    for paragraph in paragraphs:
                        # 跳过过短的段落（可能为标题、页眉页脚等）
                        if len(paragraph.strip()) < 20:
                            continue

                        # 6. 计算段落与查询的匹配分数
                        query_words = query.lower().split()  # 查询词列表（小写）
                        paragraph_lower = paragraph.lower()  # 段落文本（小写）

                        # 匹配分数 = 查询词在段落中出现的次数总和
                        score = sum(1 for word in query_words if word in paragraph_lower)

                        # 7. 更新最佳匹配结果
                        if score > best_match_score:
                            best_match_score = score
                            # 处理片段长度：超过限制则添加省略号
                            best_fragment = paragraph[:fragment_length] + "..." if len(paragraph) > fragment_length else paragraph
                            best_page_num = page_num + 1  # 页码从1开始计数

                # 8. 返回最佳匹配结果
                if best_fragment:
                    # 找到匹配片段，返回片段内容和页码
                    return best_fragment.strip(), best_page_num
                else:
                    # 9. 降级处理：未找到匹配片段时返回第一页内容
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    # 处理片段长度
                    fragment = text[:fragment_length] + "..." if len(text) > fragment_length else text
                    return fragment.strip(), 1

        except Exception as e:
            # 10. 异常处理：提取失败时返回错误信息
            print(f"从PDF提取相关片段失败 {pdf_path}: {e}")
            return "片段提取失败", "N/A"

    def classify_document(self, document_text, topics):
        """对论文进行分类"""
        if not document_text:# 内容为空，返回类型为Other
            return "Other"

        # 使用AI客户端的综合分类方法，先AI分类，如果失败，则使用关键词方法
        category = self.ai_client.classify_document(document_text, topics)
        return category

    def add_document(self, pdf_path, topics=None):
        """添加新论文并自动分类
        将PDF论文添加到系统中，自动完成以下步骤：
            1. 提取PDF文本内容
            2. 使用AI模型对论文进行分类
            3. 生成文本语义嵌入向量
            4. 复制文件到分类目录
            5. 存储到向量数据库
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return None

        if topics is None:
            topics = CATEGORIES

        filename = os.path.basename(pdf_path)

        # 提取PDF文本
        document_text = self.extract_text_from_pdf(pdf_path)
        if not document_text.strip():
            print(f"无法从PDF中提取文本: {filename}")
            return None

        # 分类论文
        category = self.classify_document(document_text, topics)

        # 生成嵌入向量
        embedding = self.ai_client.embed_text(document_text)

        # 生成唯一ID
        document_id = str(uuid.uuid4())

        # 创建新文件路径
        new_path = os.path.join(DOCUMENTS_DIR, category, filename)

        # 复制文件到分类目录
        shutil.copy2(pdf_path, new_path)

        # 存储到向量数据库
        metadata = {
            "filename": filename,
            # 分类和路径信息
            "category": category,
            "path": new_path,
            # 内容信息
            "text_preview": document_text[:1000],  # 文本预览
            # 时间信息
            "add_timestamp": time.time(),  # 添加时间戳
        }

        self.vector_db.add_document(document_id, embedding, metadata)

        print(f"论文添加成功: {filename} 的分类是 {category}")
        return {
            "id": document_id,
            "filename": filename,
            "category": category,
            "path": new_path,
            "text_preview": document_text[:200]
        }

    def organize_documents(self, source_dir, topics=None):
        """批量整理文件夹中的PDF文件"""
        if not os.path.exists(source_dir):
            print(f"源目录不存在: {source_dir}")
            return []

        if topics is None:
            topics = CATEGORIES

        results = []
        for filename in os.listdir(source_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(source_dir, filename)
                result = self.add_document(pdf_path, topics)
                if result:
                    results.append(result)
        return results

    def search_documents(self, query, top_k=TOP_K_RESULTS, simple_list=False):
        """语义搜索论文
        基于自然语言查询进行语义搜索，返回最相关的文档列表。
        支持两种返回格式：简单列表（仅基本信息）和详细列表（包含相似度、相关片段等）。
        默认返回详细列表
        """
        # 1. 查询向量化：将自然语言查询转换为语义向量
        # 使用配置的文本嵌入模型生成查询向量
        query_embedding = self.ai_client.embed_text(query)

        # 2. 向量数据库搜索：基于余弦相似度查找最相关的文档
        # 返回结果包含文档ID、元数据、相似度距离等信息
        results = self.vector_db.search_documents(query_embedding, top_k)

        # 3. 结果格式化：根据需求构建不同详细程度的返回结果
        formatted_results = []

        # 遍历搜索结果（results['ids'][0]包含top_k个结果ID）
        for i in range(len(results['ids'][0])):
            if simple_list:
                # 简化模式：仅返回基本信息，适用于快速文件列表展示
                result_info = {
                    "id": results['ids'][0][i],  # 文档唯一标识符
                    "filename": results['metadatas'][0][i]['filename'],
                    "category": results['metadatas'][0][i]['category'],
                    "path": results['metadatas'][0][i]['path'],
                    "text_preview": results['metadatas'][0][i]['text_preview'],
                }
            else:
                # 详细模式：返回完整信息，适用于搜索结果展示页面
                result_info = {
                    "id": results['ids'][0][i],  # 文档唯一标识符
                    "filename": results['metadatas'][0][i]['filename'],  # 文件名
                    "category": results['metadatas'][0][i]['category'],  # 分类结果
                    "path": results['metadatas'][0][i]['path'],  # 文件存储路径
                    "similarity": 1 - results['distances'][0][i]  # 相似度分数（1-距离）
                }

                # 4. 提取相关片段：从PDF中查找与查询最相关的文本段落
                try:
                    # 使用查询语句在文档内容中定位最相关的段落
                    relevant_fragment, page_num = self.extract_relevant_fragment(
                        results['metadatas'][0][i]['path'],  # 文档路径
                        query  # 原始查询
                    )
                    result_info["relevant_fragment"] = relevant_fragment  # 相关文本片段
                    result_info["page"] = page_num  # 片段所在页码
                except Exception as e:
                    # 片段提取失败时的降级处理
                    result_info["relevant_fragment"] = "片段提取失败"
                    result_info["page"] = "N/A"
                    # 可选：记录错误日志
                    # print(f"提取相关片段失败 {results['metadatas'][0][i]['filename']}: {e}")

            # 将格式化后的结果添加到返回列表
            formatted_results.append(result_info)

        # 5. 返回格式化后的搜索结果
        return formatted_results

    def get_simple_document_list(self, query, top_k=TOP_K_RESULTS):
        """根据查询返回简单的论文列表"""
        return self.search_documents(query, top_k, simple_list=True)

    def get_all_documents(self):
        """获取所有论文列表"""
        results = self.vector_db.get_all_documents()
        documents = []

        if 'ids' in results and results['ids']:
            for i in range(len(results['ids'])):
                documents.append({
                    "id": results['ids'][i],
                    "filename": results['metadatas'][i]['filename'],
                    "category": results['metadatas'][i]['category'],
                    "path": results['metadatas'][i]['path'],
                    "text_preview": results['metadatas'][i]['text_preview'],
                })

        return documents

    def clear_all_documents(self):
        """清空所有论文数据"""
        try:
            self.vector_db.clear_documents()

            for category in CATEGORIES:
                category_path = os.path.join(DOCUMENTS_DIR, category)
                if os.path.exists(category_path):
                    for filename in os.listdir(category_path):
                        file_path = os.path.join(category_path, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

            print("成功清空所有论文数据")
            return True
        except Exception as e:
            print(f"清空论文数据时出错: {e}")
            return False

    def delete_document(self, identifier):
        """删除指定ID或文件名的论文"""
        documents = self.get_all_documents()
        document_to_delete = None

        for document in documents:
            if document['id'] == identifier or document['filename'] == identifier:
                document_to_delete = document
                break

        if not document_to_delete:
            print(f"未找到对应论文: {identifier}")
            return False

        try:
            # 删除向量数据库中的数据
            self.vector_db.delete_document(document_to_delete['id'])
            # 删除本地文件
            if os.path.exists(document_to_delete['path']):
                os.remove(document_to_delete['path'])

            # print(f"成功删除论文: {document_to_delete['filename']}")
            return True
        except Exception as e:
            print(f"删除论文时出错: {e}")
            return False