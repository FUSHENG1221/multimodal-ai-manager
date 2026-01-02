import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import hashlib
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_CHAT_API_URL


class AIClient:
    def __init__(self, text_model_type="local", image_model_type="clip"):
        self.text_model_type = text_model_type
        self.image_model_type = image_model_type
        self.text_model = None
        self.image_model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_models()

    def _load_models(self):
        """加载指定的模型"""
        # 加载文本模型
        if self.text_model_type == "local":
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_model = self.text_model.to(self.device)
        elif self.text_model_type == "clip":
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            self.text_model = model
            self.preprocess = preprocess
        elif self.text_model_type == "multilingual":
            self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.text_model = self.text_model.to(self.device)

        # 加载图像模型
        if self.image_model_type == "clip":
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            self.image_model = model
            self.preprocess = preprocess
        elif self.image_model_type == "resnet":
            self.image_model = models.resnet50(pretrained=True)
            self.image_model = self.image_model.to(self.device)
            self.image_model.eval()
            # ResNet预处理
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def embed_text(self, text):
        """生成文本嵌入向量
        根据配置的文本模型类型，将输入文本转换为向量表示。支持CLIP模型和Sentence Transformers模型。
        Args:
            text (str): 需要转换为向量的文本内容
        Returns:
            list: 文本的向量表示，格式为浮点数列表
        Raises:
            Exception: 模型处理过程中可能出现的异常
        """
        if self.text_model_type == "clip":
            # 使用CLIP模型处理文本
            # 1. 将文本转换为CLIP模型可识别的token格式
            text_tokens = clip.tokenize([text]).to(self.device)

            # 2. 使用CLIP文本编码器生成特征向量（不计算梯度以提高效率）
            with torch.no_grad():
                text_features = self.text_model.encode_text(text_tokens)

                # 3. 对特征向量进行L2归一化，便于后续的相似度计算
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # 4. 将PyTorch张量转换为numpy数组，再转换为Python列表格式
            # squeeze()用于去除批量维度，cpu()确保数据在CPU上，numpy()转换为numpy数组
            return text_features.squeeze().cpu().numpy().tolist()
        else:
            # 使用Sentence Transformers模型处理文本
            # 1. 直接调用Sentence Transformers的encode方法生成嵌入向量
            #    该方法自动处理文本预处理、编码和归一化
            embedding = self.text_model.encode([text])

            # 2. 返回第一个（也是唯一一个）文本的嵌入向量
            #    因为输入是列表，返回的也是列表形式的嵌入向量
            return embedding[0].tolist()

    def embed_image(self, image_path):
        """生成图像嵌入向量"""
        try:
            # 1. 图像加载和预处理
            # 使用PIL库打开图像并统一转换为RGB格式，确保颜色通道一致性
            image = Image.open(image_path).convert('RGB')

            # 2. 根据配置的模型类型选择相应的处理流程
            if self.image_model_type == "clip":
                # 使用CLIP模型处理图像（适合图文跨模态检索）

                # 2.1 图像预处理：调整尺寸、归一化等操作
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # 2.2 特征提取：使用CLIP的图像编码器生成特征向量
                with torch.no_grad():  # 禁用梯度计算，提高推理速度
                    image_features = self.image_model.encode_image(image_input)

                    # 2.3 特征归一化：L2归一化，便于余弦相似度计算
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # 2.4 格式转换：PyTorch张量 → NumPy数组 → Python列表
                return image_features.squeeze().cpu().numpy().tolist()

            elif self.image_model_type == "resnet":
                # 使用ResNet模型处理图像（适合图像分类任务）

                # 2.1 图像预处理：ResNet特定的预处理流程
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # 2.2 特征提取：使用ResNet模型进行前向传播
                with torch.no_grad():  # 推理模式，不计算梯度
                    image_features = self.image_model(image_input)

                # 2.3 格式转换：直接返回模型输出（未归一化的特征向量）
                return image_features.squeeze().cpu().numpy().tolist()

        except Exception as e:
            # 3. 异常处理：模型处理失败时使用回退方法
            print(f"使用{self.image_model_type}模型处理图像失败 {image_path}: {e}")

            # 3.1 回退策略：基于图像内容的哈希嵌入
            return self.fallback_image_embedding(image_path)

    def fallback_image_embedding(self, file_path, dimension=512):
        """回退嵌入方法"""
        # 1. 读取文件内容并计算MD5哈希值
        # MD5生成128位哈希值，确保不同文件具有不同的数字签名
        with open(file_path, 'rb') as f:
            file_hash = int(hashlib.md5(f.read()).hexdigest(), 16)

        # 2. 将哈希值转换为固定维度的伪嵌入向量
        # 算法原理：通过位偏移和模运算生成数值序列
        return [float((file_hash >> i) % 1000) / 1000.0 for i in range(dimension)]

    def get_text_embedding_dimension(self):
        """获取文本嵌入维度"""
        if self.text_model_type == "clip":
            return 512
        elif self.text_model_type == "local":
            return 384
        elif self.text_model_type == "multilingual":
            return 384
        return 384

    def get_image_embedding_dimension(self):
        """获取图像嵌入维度"""
        if self.image_model_type == "clip":
            return 512
        elif self.image_model_type == "resnet":
            return 1000  # ResNet50输出1000维
        return 512

    def classify_document_by_ai(self, document_text, categories):
        """使用AI API对论文进行分类"""
        if not document_text:
            return "Other"

        # 构建分类提示
        topics_str = ", ".join(categories)
        prompt = f"""请仔细阅读以下学术论文内容，并将其分类到最合适的类别中：

        可选类别：{topics_str}

        论文内容：{document_text[:2000]}  # 限制文本长度避免超出API限制

        请严格按照以下格式直接返回分类结果：
        只返回类别名称，不要返回任何其他内容，例如：CV 或 NLP 或 RL 或 Other 等等"""

        # 尝试使用DeepSeek API进行分类
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 20
            }

            # 使用chat completions端点
            chat_api_url = DEEPSEEK_CHAT_API_URL
            response = requests.post(chat_api_url,
                                     json=data, headers=headers)
            response.raise_for_status()
            result = response.json()

            # 获取API返回的分类结果
            category = result["choices"][0]["message"]["content"].strip()

            # 验证返回的类别是否在允许的范围内
            for topic in categories:
                if topic.lower() in category.lower() or category.lower() == topic.lower():
                    return topic

            # 如果API返回的类别不在预定义范围内，返回"Other"
            return "Other"

        except Exception as e:
            print(f"AI分类API调用失败: {e}")
            # API调用失败，回退到关键词匹配分类
            return None

    def classify_document_by_keywords(self, document_text, categories):
        """基于关键词的论文分类（回退方法）"""
        if not document_text:
            return "Other"

        document_text_lower = document_text.lower()

        # 关键词映射
        keyword_mapping = {
            "CV": ["computer vision", "image", "visual", "cnn", "convolutional", "object detection",
                   "segmentation", "recognition", "optical flow", "feature extraction", "vision", "image processing"],
            "NLP": ["natural language processing", "text", "language", "transformer", "bert",
                    "token", "embedding", "linguistic", "semantic", "syntax", "nlp", "text mining"],
            "RL": ["reinforcement learning", "agent", "policy", "reward", "gym", "environment",
                   "q-learning", "markov", "dynamic programming", "rl", "reinforcement"],
            "Multimodal": ["multimodal", "cross-modal", "vision-language", "text-image",
                           "audio-visual", "multi-modal", "fusion", "multimodal learning"],
            "Autonomous Driving": ["autonomous driving", "self-driving", "adverse", "traffic",
                                   "navigation", "lidar", "radar", "automotive", "autonomous vehicle", "driving"],
            "AI": ["artificial intelligence", "ai system", "cognitive", "intelligent system", "ai",
                   "machine intelligence"],
            "ML": ["machine learning", "supervised", "unsupervised", "neural network",
                   "deep learning", "training", "inference", "ml", "machine learning"]
        }

        category_scores = {category: 0 for category in categories}

        for category, keywords in keyword_mapping.items():
            if category in categories:
                for keyword in keywords:
                    if keyword in document_text_lower:
                        category_scores[category] += 1

        # 找出得分最高的类别
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category

        return "Other"

    def classify_document(self, document_text, categories):
        """综合分类方法：先尝试AI分类，失败则使用关键词分类"""
        if not document_text:
            return "Other"

        # 首先尝试AI分类
        ai_category = self.classify_document_by_ai(document_text, categories)

        # 如果AI分类成功且不是"Other"，直接返回
        if ai_category and ai_category != "Other":
            return ai_category

        # 如果AI分类失败或返回"Other"，使用关键词分类
        return self.classify_document_by_keywords(document_text, categories)