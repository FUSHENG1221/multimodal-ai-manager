import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# 分类目录
CATEGORIES = ["CV", "NLP", "RL", "Multimodal", "Autonomous Driving", "AI", "ML", "Other"]
for category in CATEGORIES:
    os.makedirs(os.path.join(DOCUMENTS_DIR, category), exist_ok=True)

# 向量数据库目录
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

# 模型配置 - 支持多种模型配置
MODEL_CONFIGS = {
    # 文本嵌入模型配置
    "text_embedding": {
        "local": {
            "model_name": "all-MiniLM-L6-v2",
            "provider": "sentence_transformers",
            "dimension": 384
        },
        "clip": {
            "model_name": "ViT-B-32",
            "provider": "clip",
            "dimension": 512
        },
        "multilingual": {
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "provider": "sentence_transformers",
            "dimension": 384
        }
    },
    # 图像嵌入模型配置
    "image_embedding": {
        "clip": {
            "model_name": "ViT-B-32",
            "provider": "clip",
            "dimension": 512
        },
        "resnet": {
            "model_name": "resnet50",
            "provider": "torchvision",
            "dimension": 2048
        }
    }
}

# 默认模型选择
DEFAULT_TEXT_MODEL = "local"
DEFAULT_IMAGE_MODEL = "clip"

# API配置（保留用于未来扩展）
DEEPSEEK_API_KEY = "sk-8806cb3516b745f0afe9de92fcba7054"
DEEPSEEK_EMBEDDING_API_URL = "https://api.deepseek.com/v1/embeddings"
DEEPSEEK_CHAT_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 搜索配置
TOP_K_RESULTS = 5
IMAGE_TOP_K_RESULTS = 3

# 确保所有目录存在
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)