# run_native_rag.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag import NativeRAGRetriever, query_to_text, generate_analysis
from config import EMBEDDING_MODEL, EVENTS_DIR, EMBEDDINGS_PATH, get_client


def main():
    # 1. 初始化 Embedder
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 2. 初始化检索器
    retriever = NativeRAGRetriever(EVENTS_DIR, embedder)

    # 可选：保存嵌入，下次直接加载
    # retriever.save_embeddings(EMBEDDINGS_PATH)
    # retriever.load_embeddings(EMBEDDINGS_PATH)

    # 3. 测试查询
    test_queries = {
        "event_name": "美伊战争",
        "event_type": "地缘冲突",
        "subtype": "战争",
        "countries": ["伊朗", "美国"],
        "indicators": {"库存": "低位"}
    }

    results = retriever.search(test_queries, top_k=5)

    print(f"\n📊 检索结果:")
    for j, r in enumerate(results, 1):
        print(f"  {j}. {r['event']['event_name']} (相似度: {r['similarity']:.3f})")

    # 获取 client 并生成分析
    client = get_client()  # 从配置获取
    analysis = generate_analysis(test_queries, results, client)
    print(f"\n📝 分析报告:\n{analysis}")


if __name__ == "__main__":
    main()