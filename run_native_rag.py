# run_native_rag.py
import numpy as np
from pathlib import Path
from rag import NativeRAGRetriever, query_to_text
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "F:/deep learning/bge-large-zh-v1.5"
LLM_MODEL = r"F:\deep learning\qwen\Qwen2___5-7B-Instruct"

def main():
    # 1. 初始化 Embedder
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 2. 初始化检索器
    events_dir = "output/"  # 你的事件 JSON 文件夹
    retriever = NativeRAGRetriever(events_dir, embedder)

    # 可选：保存嵌入，下次直接加载
    # retriever.save_embeddings("output/event_embeddings.pkl")
    # retriever.load_embeddings("output/event_embeddings.pkl")

    # 3. 测试查询
    test_queries = [
        {
            "event_type": "地缘冲突",
            "subtype": "战争",
            "countries": ["伊朗", "美国"],
            "indicators": {"库存": "低位"}
        },
        {
            "event_type": "地缘冲突",
            "subtype": "战争",
            "countries": ["俄罗斯", "乌克兰"]
        },
        {
            "event_type": "供给冲击",
            "subtype": "减产",
            "indicators": {"OPEC产能": "紧张"}
        },
        {
            "query_text": "美联储加息对油价的影响"
        }
    ]

    # 4. 运行检索
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {query_to_text(query)}")
        print('=' * 60)

        results = retriever.search(query, top_k=5)

        print(f"\n📊 检索结果:")
        for j, r in enumerate(results, 1):
            print(f"  {j}. {r['event']['event_name']} (相似度: {r['similarity']:.3f})")

        # 可选：生成 LLM 分析
        # analysis = generate_analysis(query, results, llm_client)
        # print(f"\n📝 分析报告:\n{analysis}")


if __name__ == "__main__":
    main()