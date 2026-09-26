md5_path = "./md5.text"

# Chroma向量库配置
collection_name = "sql_knowledge_db"  # 从默认"rag"改成场景化库名
persist_directory = "./chroma_db"

# 文本切分参数：数据库知识点偏概念，调小块大小，增加重叠，保证语义完整
#chunk_size = 800
#chunk_overlap = 150
#separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
chunk_size = 500
chunk_overlap = 80
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]



max_split_char_number = 800

# 检索配置：从只返回1条改成返回3条，提升回答完整性
similarity_threshold = 3

# 模型配置
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
