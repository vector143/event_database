# config.py
import os
from openai import OpenAI

# API 配置
OPENAI_API_KEY = 'sk-e789d6fbbda54f948eee266408c73186'
OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# 模型配置
EMBEDDING_MODEL = "F:/deep learning/bge-large-zh-v1.5"
LLM_MODEL = r"F:\deep learning\qwen\Qwen2___5-7B-Instruct"

# 路径配置
EVENTS_DIR = "output/"
EMBEDDINGS_PATH = "output/event_embeddings.pkl"

# 初始化全局 client
def get_client():
    """获取 OpenAI 客户端实例"""
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

# 创建全局 client 实例（可选）
client = get_client()