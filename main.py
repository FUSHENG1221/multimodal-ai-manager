# ==================== 完全禁用所有TensorFlow日志 ====================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 在所有导入之前禁用所有警告
import warnings
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 禁用TensorFlow的日志
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except:
    pass

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
except:
    pass

import argparse
from document_manager import DocumentManager
from image_manager import ImageManager
from config import *
import sys


def get_system_info():
    """获取系统信息"""
    info = {
        "系统信息": {
            "项目名称": "多模态AI文献与图像管理助手",
            "Python版本": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "系统平台": sys.platform
        },
        "目录结构": {
            "项目根目录": PROJECT_ROOT,
            "数据目录": DATA_DIR,
            "文献目录": DOCUMENTS_DIR if 'DOCUMENTS_DIR' in globals() else DOCUMENTS_DIR,
            "图片目录": IMAGES_DIR,
            "向量数据库": VECTOR_DB_DIR,
        },
        "模型配置": {
            "默认文本模型": DEFAULT_TEXT_MODEL,
            "默认图片模型": DEFAULT_IMAGE_MODEL,
            "分类主题": CATEGORIES
        }
    }
    return info


def get_statistics():
    """获取文献和图片统计信息"""
    stats = {"文献统计": {}, "图片统计": {}}

    try:
        # 获取文献统计
        doc_manager = DocumentManager()
        documents = doc_manager.get_all_documents()
        stats["文献统计"]["总数"] = len(documents)

        # 按分类统计文献
        category_stats = {}
        for doc in documents:
            category = doc.get('category', 'Other')
            category_stats[category] = category_stats.get(category, 0) + 1

        # 确保所有分类都显示，即使是0
        for category in CATEGORIES:
            if category not in category_stats:
                category_stats[category] = 0

        stats["文献统计"]["分类统计"] = category_stats

    except Exception as e:
        stats["文献统计"]["错误"] = f"获取文献统计失败: {e}"

    try:
        # 获取图片统计
        img_manager = ImageManager()
        images = img_manager.get_all_images()
        stats["图片统计"]["总数"] = len(images)

        # 按扩展名统计图片
        extension_stats = {}
        for img in images:
            filename = img.get('filename', '')
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                extension_stats[ext] = extension_stats.get(ext, 0) + 1

        stats["图片统计"]["格式统计"] = extension_stats

    except Exception as e:
        stats["图片统计"]["错误"] = f"获取图片统计失败: {e}"

    return stats


def display_system_info():
    """显示系统信息"""
    info = get_system_info()
    stats = get_statistics()

    # 显示系统信息
    print("🤖 系统配置:")
    for section, data in info.items():
        print(f"\n  {section}:")
        for key, value in data.items():
            if value is not None:  # 跳过None值
                print(f"    {key}: {value}")

    # 显示统计信息
    print("\n📈 统计信息:")

    # 文献统计
    print(f"\n  📚 文献管理:")
    print(f"    总文献数: {stats['文献统计'].get('总数', 0)}")
    if '分类统计' in stats['文献统计']:
        print(f"    分类统计:")
        for category, count in stats['文献统计']['分类统计'].items():
            print(f"      {category}: {count} 篇")

    # 图片统计
    print(f"\n  🖼️ 图片管理:")
    print(f"    总图片数: {stats['图片统计'].get('总数', 0)}")
    if '格式统计' in stats['图片统计']:
        print(f"    格式统计:")
        for ext, count in stats['图片统计']['格式统计'].items():
            print(f"      {ext.upper()}: {count} 张")

    # 数据库信息
    print(f"\n  💾 数据库:")
    print(f"    向量数据库: {VECTOR_DB_DIR}")
    import os
    if os.path.exists(VECTOR_DB_DIR):
        try:
            db_size = sum(os.path.getsize(os.path.join(VECTOR_DB_DIR, f))
                          for f in os.listdir(VECTOR_DB_DIR)
                          if os.path.isfile(os.path.join(VECTOR_DB_DIR, f)))
            print(f"    数据库大小: {db_size / 1024 / 1024:.2f} MB")
        except:
            print(f"    数据库大小: 未知")
    else:
        print(f"    数据库状态: 未创建")

    # 磁盘空间信息
    print(f"\n  💿 存储空间:")
    for dir_name, dir_path in [("文献目录", DOCUMENTS_DIR),
                               ("图片目录", IMAGES_DIR),
                               ("数据目录", DATA_DIR)]:
        if os.path.exists(dir_path):
            dir_size = 0
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    try:
                        dir_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            print(f"    {dir_name}: {dir_size / 1024 / 1024:.2f} MB")
        else:
            print(f"    {dir_name}: 目录不存在")

def main():
    parser = argparse.ArgumentParser(description="万倩本地多模态AI代理 - 文献和图像管理")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 论文管理命令
    add_document_parser = subparsers.add_parser("add_document", help="添加新论文并自动分类")
    add_document_parser.add_argument("path", help="PDF文件路径，如果存在空格，请用双引号包裹")
    add_document_parser.add_argument("--topics", type=str, default=",".join(CATEGORIES),
                                  help="指定分类主题列表（逗号分隔）")
    add_document_parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL,
                                  help="文本模型类型: local, clip, multilingual")# 默认：all-MiniLM-L6-v2

    search_document_parser = subparsers.add_parser("search_document", help="语义搜索论文")
    search_document_parser.add_argument("query", help="搜索查询")
    search_document_parser.add_argument("--simple", action="store_true", help="是否简单查询")
    search_document_parser.add_argument("--top_k", type=int, default=TOP_K_RESULTS, help="返回结果数量")
    search_document_parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL,
                                     help="文本模型类型: local, clip, multilingual")

    organize_documents_parser = subparsers.add_parser("organize_documents", help="批量添加论文并自动分类")
    organize_documents_parser.add_argument("directory", help="包含论文的目录路径")
    organize_documents_parser.add_argument("--topics", type=str, default=",".join(CATEGORIES),
                                       help="分类主题列表（逗号分隔）")
    organize_documents_parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL,
                                       help="文本模型类型: local, clip, multilingual")

    list_documents_parser = subparsers.add_parser("list_documents", help="列出所有已整理的论文")
    list_documents_parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL,
                                    help="文本模型类型: local, clip, multilingual")

    delete_document_parser = subparsers.add_parser("delete_document", help="删除指定ID或文件名的论文")
    delete_document_parser.add_argument("identifier", help="论文ID或文件名")
    delete_document_parser.add_argument("--model", type=str, default=DEFAULT_TEXT_MODEL,
                                     help="文本模型类型: local, clip, multilingual")

    # 图像管理命令
    add_image_parser = subparsers.add_parser("add_image", help="添加新图像")
    add_image_parser.add_argument("path", help="图像文件路径")
    add_image_parser.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                                  help="图像模型类型: clip, resnet")

    batch_add_images_parser = subparsers.add_parser("batch_add_images", help="批量添加图像")
    batch_add_images_parser.add_argument("directory", help="包含图像的目录路径")
    batch_add_images_parser.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                                         help="图像模型类型: clip, resnet")

    search_image_parser = subparsers.add_parser("search_image", help="以文搜图")
    search_image_parser.add_argument("query", help="搜索查询")
    search_image_parser.add_argument("--top_k", type=int, default=IMAGE_TOP_K_RESULTS, help="返回结果数量")
    search_image_parser.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                                     help="图像模型类型: clip, resnet")

    list_images_parser = subparsers.add_parser("list_images", help="列出所有已存储的图像")
    list_images_parser.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                                    help="图像模型类型: clip, resnet")

    delete_image_parser = subparsers.add_parser("delete_image", help="删除指定ID或文件名的图像")
    delete_image_parser.add_argument("identifier", help="图像ID或文件名")
    delete_image_parser.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                                     help="图像模型类型: clip, resnet")

    # 系统命令
    format_parser = subparsers.add_parser("format", help="格式化整个系统")

    info_parser = subparsers.add_parser("info", help="查看系统信息")

    args = parser.parse_args()

    # 1、添加新论文并自动分类
    if args.command == "add_document":
        doc_manager = DocumentManager(text_model_type=args.model)
        topics = [topic.strip() for topic in args.topics.split(",")]
        result = doc_manager.add_document(args.path, topics)
        if result:
            print(f"\n论文添加后信息:")
            print(f"ID: {result['id']}")
            print(f"文件名: {result['filename']}")
            print(f"分类: {result['category']}")
            print(f"路径: {result['path']}")
            print(f"预览: {result['text_preview']}")

    # 2、语义搜索论文,支持简单搜索和复杂搜索，复杂搜索会输出相关片段
    elif args.command == "search_document":
        doc_manager = DocumentManager(text_model_type=args.model)
        results = doc_manager.search_documents(args.query, args.top_k,simple_list=args.simple)

        if results:
            print(f"找到 {len(results)} 篇相关论文:")
            for i, result in enumerate(results, 1):
                print(f"第{i}篇： {result['filename']}")
                print(f"   ID: {result['id']}")
                print(f"   分类: {result['category']}")
                print(f"   路径: {result['path']}")
                if not args.simple:
                    print(f"   相似度: {result['similarity']:.4f}")
                    print(f"   相关片段: {result['relevant_fragment']}")
                    print(f"   相关片段页码: {result['page']}")
        else:
            print("未找到相关论文。")

    # 3、批量添加论文并自动分类
    elif args.command == "organize_documents":
        doc_manager = DocumentManager(text_model_type=args.model)
        topics = [topic.strip() for topic in args.topics.split(",")]
        results = doc_manager.organize_documents(args.directory, topics)
        print(f"批量整理完成,已处理 {len(results)} 篇论文。")

    # 4、列出所有已整理的论文
    elif args.command == "list_documents":
        doc_manager = DocumentManager(text_model_type=args.model)
        documents = doc_manager.get_all_documents()
        print(f"目前系统已整理 {len(documents)} 篇论文:")
        for i, document in enumerate(documents, 1):
            print(f"第{i}篇: {document['filename']}")
            print(f"   ID: {document['id']}")
            print(f"   分类: {document['category']}")
            # print(f"   路径: {document['path']}")
            # print(f"   预览: {document['text_preview']}")

    # 5、删除指定ID或文件名的论文
    elif args.command == "delete_document":
        doc_manager = DocumentManager(text_model_type=args.model)
        success = doc_manager.delete_document(args.identifier)
        print(f"论文删除{'成功' if success else '失败'}: {args.identifier}")

    # 6、添加图像
    elif args.command == "add_image":
        img_manager = ImageManager(image_model_type=args.model)
        result = img_manager.add_image(args.path)
        if result:
            print(f"图像添加成功:")
            print(f"    ID: {result['id']}")
            print(f"    文件名: {result['filename']}")
            print(f"    路径: {result['path']}")

    # 7、批量添加图像
    elif args.command == "batch_add_images":
        img_manager = ImageManager(image_model_type=args.model)
        results = img_manager.batch_add_images(args.directory)
        print(f"批量导入完成,已成功导入 {len(results)} 张图像。")

    # 8、以文搜图
    elif args.command == "search_image":
        img_manager = ImageManager(image_model_type=args.model)
        results = img_manager.search_images(args.query, args.top_k)
        print(f"找到 {len(results)} 张相关图像:")
        for i, result in enumerate(results, 1):
            print(f"第{i}张: {result['filename']}")
            print(f"   ID: {result['id']}")
            print(f"   路径: {result['path']}")
            print(f"   相似度: {result['similarity']:.4f}")

    # 9、列出所有已存储的图像
    elif args.command == "list_images":
        img_manager = ImageManager(image_model_type=args.model)
        images = img_manager.get_all_images()
        print(f"\n已存储 {len(images)} 张图像:")
        for i, image in enumerate(images, 1):
            print(f"第{i}张: {image['filename']}")
            print(f"   ID: {image['id']}")
            print(f"   路径: {image['path']}")

    # 10、删除指定ID或文件名的图像
    elif args.command == "delete_image":
        img_manager = ImageManager(image_model_type=args.model)
        success = img_manager.delete_image(args.identifier)
        print(f"图像删除{'成功' if success else '失败'}: {args.identifier}")

    # 11、格式化整个系统
    elif args.command == "format":
        print("正在格式化整个系统...")
        confirmation = input("确定要继续吗？这将清空所有文献和图片数据。请输入yes或者no: ")
        if confirmation.lower() == "yes":
            doc_manager = DocumentManager()
            img_manager = ImageManager()
            doc_manager.clear_all_documents()
            img_manager.clear_all_images()
            print("格式化完成！")
        else:
            print("操作已取消。")

    # 12、查看系统信息
    elif args.command == "info":
        display_system_info()


    else:
        parser.print_help()


if __name__ == "__main__":
    main()