import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import pickle


class NativeRAGRetriever:
    def __init__(self, events_dir: str, embedder):
        """
        events_dir: 存放事件 JSON 的文件夹
        embedder: EventEmbedder 实例
        """
        self.embedder = embedder
        self.events = []
        self.embeddings = []
        self.event_ids = []
        self._load_events(events_dir)

    def _load_events(self, events_dir: str):
        """加载所有事件并生成嵌入"""
        events_path = Path(events_dir)
        event_files = list(events_path.glob("*.json"))

        # 排除汇总文件
        event_files = [f for f in event_files if f.name != "_all_events.json"]

        print(f"📂 加载 {len(event_files)} 个事件文件...")

        for file in event_files:
            with open(file, 'r', encoding='utf-8') as f:
                event = json.load(f)
                self.events.append(event)
                self.event_ids.append(event.get('event_id', file.stem))

        # 批量生成嵌入
        print(f"🔢 生成 {len(self.events)} 个事件的嵌入...")
        texts = [build_event_text(e) for e in self.events]

        # 打印示例
        print(f"📝 文本示例:\n{texts[0][:200]}...\n")

        self.embeddings = self.embedder.encode(texts)
        print(f"✅ 嵌入完成，维度: {self.embeddings[0].shape}")

    def search(self, query: dict, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相似事件

        Parameters:
        -----------
        query: 结构化查询，格式如:
            {
                "event_type": "地缘冲突",
                "subtype": "战争",
                "countries": ["伊朗", "美国"],
                "indicators": {"库存": "低位"}
            }
        top_k: 返回数量

        Returns:
        --------
        list of dict: 每个包含 event, similarity, event_id
        """
        # 1. 查询转文本
        query_text = query_to_text(query)
        print(f"🔍 查询文本: {query_text}")

        # 2. 查询向量化
        query_emb = self.embedder.encode([query_text])[0]

        # 3. 计算余弦相似度（向量已归一化，点积=余弦）
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb)
            similarities.append(sim)

        # 4. 取 top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "event": self.events[idx],
                "event_id": self.event_ids[idx],
                "similarity": float(similarities[idx])
            })

        return results

    def save_embeddings(self, save_path: str):
        """保存嵌入和事件数据，避免重复生成"""
        data = {
            'events': self.events,
            'event_ids': self.event_ids,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'model': str(self.embedder.model),
            'n_events': len(self.events)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 嵌入已保存到 {save_path}")

    def load_embeddings(self, load_path: str):
        """加载已保存的嵌入"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.events = data['events']
        self.event_ids = data['event_ids']
        self.embeddings = [np.array(emb) for emb in data['embeddings']]
        print(f"📂 已加载 {len(self.events)} 个事件的嵌入")


def build_event_text(event: dict) -> str:
    """
    从事件 JSON 构建用于向量化的文本
    不含价格路径 pattern（留给未来图检索）
    """
    parts = []

    # 核心信息
    if event.get('event_name'):
        parts.append(event['event_name'])

    if event.get('brief_description'):
        parts.append(event['brief_description'])

    # 类型信息
    event_type = event.get('event_type', '')
    subtype = event.get('subtype', '')
    if event_type or subtype:
        parts.append(f"类型：{event_type}{subtype}")

    # 国家信息
    countries = event.get('involved_countries', [])
    if countries:
        parts.append(f"涉及国家：{', '.join(countries)}")

    # 总结
    if event.get('summary'):
        parts.append(event['summary'])

    return "。".join(parts)


def query_to_text(query: dict) -> str:
    """
    结构化查询转自然语言文本
    """
    parts = []

    if query.get('event_type'):
        subtype = query.get('subtype', '')
        parts.append(f"事件类型：{query['event_type']}{subtype}")

    if query.get('countries'):
        parts.append(f"涉及国家：{', '.join(query['countries'])}")

    if query.get('indicators'):
        ind_str = '、'.join([f"{k}{v}" for k, v in query['indicators'].items()])
        parts.append(f"指标状态：{ind_str}")

    # 如果没有结构化字段，直接用文本
    if not parts and query.get('query_text'):
        return query['query_text']

    return "，".join(parts)


def generate_analysis(query: dict, results: List[Dict], llm_client) -> str:
    """用 LLM 生成对比分析报告"""

    # 构建事件对比文本
    events_text = ""
    for i, r in enumerate(results, 1):
        event = r["event"]
        sim = r["similarity"]

        # 提取价格路径关键数据
        price = event.get('price_path', {}).get('brent', {})
        peak = price.get('peak', {})
        trough = price.get('trough', {})

        events_text += f"""
### 事件{i}: {event['event_name']} (相似度: {sim:.3f})
- 类型: {event['event_type']}/{event.get('subtype', '')}
- 涉及国家: {', '.join(event.get('involved_countries', []))}
- 核心描述: {event.get('brief_description', '')}
- 价格峰值: {peak.get('value', 'N/A')}美元/桶 (涨幅 {peak.get('return_since_pre', 'N/A')}%)
- 价格谷值: {trough.get('value', 'N/A')}美元/桶
- 价格模式: {price.get('pattern', '未标注')}
- 总结: {event.get('summary', '')[:100]}...
"""

    prompt = f"""
请根据以下历史事件数据，分析当前查询的参考价值。

【当前查询】
{json.dumps(query, ensure_ascii=False, indent=2)}

【相似历史事件】
{events_text}

请分析：
1. 这些历史事件与当前查询的相似点是什么？
2. 根据历史经验，类似事件通常对油价有什么影响？（关注涨幅、持续时间、波动率）
3. 结合当前查询的特点，给出判断建议。

只输出分析结论，不要输出JSON。
"""

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个原油市场分析专家，擅长基于历史事件进行类比分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 分析失败: {e}")
        return "分析生成失败"