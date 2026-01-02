import os
import shutil
import time
import uuid
from PIL import Image
from config import *
from ai_client import AIClient
from vector_db import VectorDB


class ImageManager:
    def __init__(self, image_model_type=DEFAULT_IMAGE_MODEL):
        self.image_model_type = image_model_type
        self.ai_client = AIClient(image_model_type=image_model_type)
        self.vector_db = VectorDB(image_dimension=self.ai_client.get_image_embedding_dimension())

    def is_valid_image(self, file_path):
        """检查文件是否为有效的图像"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except:
            return False

    def add_image(self, image_path):
        """添加新图像到系统"""
        # 1. 验证文件存在性
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return None

        # 2. 验证图像文件有效性（格式、完整性等）
        if not self.is_valid_image(image_path):
            print(f"无效的图像文件: {image_path}")
            return None

        # 3. 提取文件名（不包含路径）
        filename = os.path.basename(image_path)

        # 4. 生成图像的语义嵌入向量
        # 使用配置的图像模型（CLIP/ResNet等）将图像转换为特征向量
        embedding = self.ai_client.embed_image(image_path)

        # 5. 生成全局唯一标识符（UUID）
        # 确保每个图像在系统中的唯一性，避免ID冲突
        image_id = str(uuid.uuid4())

        # 6. 构建图像在系统中的存储路径
        new_path = os.path.join(IMAGES_DIR, filename)

        # 7. 复制图像文件到系统目录（避免重复复制）
        # 使用copy2保留文件的元数据（创建时间、修改时间等）
        if not os.path.exists(new_path):
            shutil.copy2(image_path, new_path)
            # print(f"图像文件已复制到: {new_path}")

        # 8. 构建图像的元数据信息
        metadata = {
            "filename": filename,
            "path": new_path,
            "add_timestamp": time.time(),  # 添加时间戳
        }

        self.vector_db.add_image(image_id, embedding, metadata)

        # print(f"图像添加成功: {filename}")
        return {
            "id": image_id,
            "filename": filename,
            "path": new_path
        }

    def batch_add_images(self, source_dir):
        """批量添加目录中的所有图像"""
        if not os.path.exists(source_dir):
            print(f"源目录不存在: {source_dir}")
            return []

        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    result = self.add_image(file_path)
                    if result:
                        results.append(result)

        return results

    def search_images(self, text_query, top_k=5):
        """以文搜图：根据文本描述搜索相似图像"""
        query_embedding = self.ai_client.embed_text(text_query)
        results = self.vector_db.search_images(query_embedding, top_k)

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "filename": results['metadatas'][0][i]['filename'],
                "path": results['metadatas'][0][i]['path'],
                "similarity": 1 - results['distances'][0][i]
            })

        return formatted_results

    def get_all_images(self):
        """获取所有图像列表"""
        results = self.vector_db.get_all_images()
        images = []

        if 'ids' in results and results['ids']:
            for i in range(len(results['ids'])):
                images.append({
                    "id": results['ids'][i],
                    "filename": results['metadatas'][i]['filename'],
                    "path": results['metadatas'][i]['path']
                })

        return images

    def clear_all_images(self):
        """清空所有图像数据"""
        try:
            self.vector_db.clear_images()

            if os.path.exists(IMAGES_DIR):
                for filename in os.listdir(IMAGES_DIR):
                    file_path = os.path.join(IMAGES_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            print("成功清空所有图像数据")
            return True
        except Exception as e:
            print(f"清空图像数据时出错: {e}")
            return False

    def delete_image(self, identifier):
        """删除指定ID或文件名的图像"""
        images = self.get_all_images()
        image_to_delete = None

        for image in images:
            if image['id'] == identifier or image['filename'] == identifier:
                image_to_delete = image
                break

        if not image_to_delete:
            print(f"未找到图像: {identifier}")
            return False

        try:
            # 从向量数据库中删除图像
            self.vector_db.delete_image(image_to_delete['id'])
            # 从文件系统中删除图像文件
            if os.path.exists(image_to_delete['path']):
                os.remove(image_to_delete['path'])

            print(f"成功删除图像: {image_to_delete['filename']}")
            return True
        except Exception as e:
            print(f"删除图像时出错: {e}")
            return False