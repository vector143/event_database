import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any, Union
import openai
import os
from pathlib import Path
import glob
from openai import OpenAI


os.environ['OPENAI_API_KEY'] = 'sk-e789d6fbbda54f948eee266408c73186'


class HistoricalEventDatabaseBuilder:
    def __init__(self,
                 data_folder: str,
                 price_df: pd.DataFrame = None,
                 api_key: str = None):
        """
        初始化数据库构建器

        Parameters:
        -----------
        data_folder : str
            存放所有指标CSV文件的文件夹路径
        price_df : pd.DataFrame
            包含布伦特和WTI价格的时间序列数据，索引为日期（可选，也可从CSV加载）
        api_key : str
            LLM API密钥
        """
        self.data_folder = Path(data_folder)
        self.price_df = price_df
        self.api_key = api_key
        self.indicators_cache = {}  # 缓存已加载的指标数据

        if api_key:
            openai.api_key = api_key

        # 自动加载所有指标数据
        self.load_all_indicators()

    def load_indicator_from_csv(self, file_path: str,
                                date_column: str = 'date',
                                value_column: str = 'value',
                                freq: str = None) -> pd.Series:
        """
        从CSV文件加载单个指标的时间序列

        Parameters:
        -----------
        file_path : str
            CSV文件路径
        date_column : str
            日期列名
        value_column : str
            数值列名
        freq : str
            数据频率（'D'=日度, 'M'=月度, 'Q'=季度, 'Y'=年度）
            如果为None，则自动推断
        """
        try:
            df = pd.read_csv(file_path)

            # 转换日期列
            df[date_column] = pd.to_datetime(df[date_column])

            # 设置索引
            df = df.set_index(date_column)

            # 提取数值列
            series = df[value_column].sort_index()

            # 如果指定了频率，设置为相应的频率
            if freq:
                series = series.asfreq(freq)

            # 存储元数据
            series.attrs['freq'] = freq or self._infer_frequency(series)
            series.attrs['file'] = file_path

            return series

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return None

    def _infer_frequency(self, series: pd.Series) -> str:
        """自动推断数据频率"""
        if len(series) < 2:
            return 'unknown'

        # 计算日期差
        date_diff = series.index.to_series().diff().dropna()
        median_diff = date_diff.median()

        # 根据天数判断频率
        if median_diff <= timedelta(days=1.5):
            return 'D'  # 日度
        elif median_diff <= timedelta(days=35):
            return 'M'  # 月度
        elif median_diff <= timedelta(days=100):
            return 'Q'  # 季度
        elif median_diff <= timedelta(days=400):
            return 'Y'  # 年度
        else:
            return 'unknown'

    def load_all_indicators(self, config: Dict[str, Dict] = None):
        """
        加载所有指标数据

        Parameters:
        -----------
        config : Dict
            指标配置，格式：
            {
                'inventory': {'file': 'inventory.csv', 'date_col': 'date', 'value_col': 'value', 'freq': 'M'},
                'opec_spare': {'file': 'opec_spare.csv', 'date_col': 'date', 'value_col': 'value', 'freq': 'M'},
                ...
            }
            如果为None，则自动扫描文件夹中的CSV文件
        """
        if config is None:
            # 自动扫描CSV文件
            csv_files = glob.glob(str(self.data_folder / "*.csv"))
            for file_path in csv_files:
                file_name = Path(file_path).stem
                # 尝试加载，假设文件名为指标名称
                series = self.load_indicator_from_csv(file_path)
                if series is not None:
                    self.indicators_cache[file_name] = series
                    print(f"已加载指标: {file_name}, 频率: {series.attrs['freq']}, 数据点: {len(series)}")
        else:
            # 按配置加载
            for indicator_name, cfg in config.items():
                file_path = self.data_folder / cfg['file']
                series = self.load_indicator_from_csv(
                    file_path,
                    date_column=cfg.get('date_col', 'date'),
                    value_column=cfg.get('value_col', 'value'),
                    freq=cfg.get('freq', None)
                )
                if series is not None:
                    self.indicators_cache[indicator_name] = series
                    print(f"已加载指标: {indicator_name}, 频率: {series.attrs['freq']}, 数据点: {len(series)}")

    def load_price_data(self,
                        brent_file: str = None,
                        wti_file: str = None,
                        date_column: str = 'date',
                        value_column: str = 'value'):
        """
        从CSV文件加载价格数据

        Parameters:
        -----------
        brent_file : str
            布伦特价格CSV文件路径
        wti_file : str
            WTI价格CSV文件路径
        date_column : str
            日期列名
        """
        price_data = {}

        if brent_file:
            brent_df = pd.read_csv(brent_file)
            brent_df[date_column] = pd.to_datetime(brent_df[date_column])
            price_data['brent'] = brent_df.set_index(date_column)[value_column].sort_index()

        if wti_file:
            wti_df = pd.read_csv(wti_file)
            wti_df[date_column] = pd.to_datetime(wti_df[date_column])
            price_data['wti'] = wti_df.set_index(date_column)[value_column].sort_index()

        self.price_df = pd.DataFrame(price_data)

    def get_value_at_date(self, series: pd.Series, target_date: pd.Timestamp,
                          method: str = 'forward') -> Optional[float]:
        """
        获取指定日期的值（处理不同频率）

        Parameters:
        -----------
        series : pd.Series
            指标时间序列
        target_date : pd.Timestamp
            目标日期
        method : str
            获取方法：
            - 'forward': 向前填充（获取最近的过去值）- 默认，适合分析背景
            - 'backward': 向后填充（获取最近的未来值）- 适合预测
            - 'nearest': 获取最近的值（前后都可）
        """
        if len(series) == 0:
            return None

        freq = series.attrs.get('freq', 'unknown')

        # ========== 月度数据特殊处理 ==========
        # 月度数据只取历史值，不取未来值
        if freq == 'M':
            if method == 'forward':
                # 只取 ≤ target_date 的历史数据
                past_dates = series.index[series.index <= target_date]
                if len(past_dates) > 0:
                    return series.loc[past_dates[-1]]
                # 如果没有历史数据，返回 None
                return None

            elif method == 'backward':
                # 只取 ≥ target_date 的未来数据
                future_dates = series.index[series.index >= target_date]
                if len(future_dates) > 0:
                    return series.loc[future_dates[0]]
                return None

            elif method == 'nearest':
                # 找到最接近的日期（不区分前后）
                idx = series.index.get_indexer([target_date], method='nearest')[0]
                if idx >= 0 and idx < len(series):
                    return series.iloc[idx]
                return None

        # ========== 日度及其他频率数据 ==========
        if method == 'forward':
            # 向前填充：取 ≤ target_date 的最新值
            if target_date in series.index:
                return series.loc[target_date]
            past_dates = series.index[series.index <= target_date]
            if len(past_dates) > 0:
                return series.loc[past_dates[-1]]
            return None

        elif method == 'backward':
            # 向后填充：取 ≥ target_date 的最早值
            if target_date in series.index:
                return series.loc[target_date]
            future_dates = series.index[series.index >= target_date]
            if len(future_dates) > 0:
                return series.loc[future_dates[0]]
            return None

        elif method == 'nearest':
            # 最近填充：取最接近的值
            if target_date in series.index:
                return series.loc[target_date]
            idx = series.index.get_indexer([target_date], method='nearest')[0]
            if idx >= 0 and idx < len(series):
                return series.iloc[idx]
            return None

        return None

    def calculate_percentile(self, series: pd.Series, value: float,
                             lookback_years: int = 5) -> Optional[float]:
        """根据频率优化分位数计算"""
        if pd.isna(value) or len(series) == 0:
            return None

        freq = series.attrs.get('freq', 'unknown')
        current_date = series.index.max()

        # 根据不同频率选择窗口
        if freq == 'M':
            # 月度数据：精确5年 = 60个月
            lookback_date = current_date - pd.DateOffset(years=lookback_years)
        else:
            # 日度数据：使用实际天数
            lookback_date = current_date - timedelta(days=lookback_years * 365)

        historical_data = series[series.index >= lookback_date].dropna()

        if len(historical_data) == 0:
            return None

        percentile = (historical_data < value).sum() / len(historical_data) * 100
        return round(percentile, 1)

    def interpret_percentile(self, percentile: float,
                             thresholds: Dict[str, float] = None) -> str:
        """根据百分位数给出解释"""
        if thresholds is None:
            thresholds = {'low': 25, 'medium': 75}

        if percentile is None:
            return "数据不足"
        elif percentile <= thresholds['low']:
            return "低位"
        elif percentile <= thresholds['medium']:
            return "中位"
        else:
            return "高位"

    def get_indicators_at_date(self, event_date: str,
                               method: str = 'forward') -> Dict[str, Any]:
        """
        获取事件发生日的所有指标数据
        """
        date = pd.to_datetime(event_date)
        indicators = {}

        # 指标单位配置
        units = {
            'inventory': '千桶',
            'opec_spare': '万桶/日',
            'brent_price': '美元/桶',
            'dxy': '',
            'vix': '',
            'refinery_util': '%'
        }

        # 指标名称映射（缓存key -> 显示名称）
        name_mapping = {
            'inventory': '美国商业原油库存',
            'opec_spare': 'OPEC闲置产能',
            'brent_price': '布伦特油价',
            'dxy': '美元指数',
            'vix': 'VIX恐慌指数',
            'refinery_util': '美国炼厂开工率'
        }

        for indicator_name, series in self.indicators_cache.items():
            # 获取指定日期的值
            value = self.get_value_at_date(series, date, method=method)

            if value is not None:
                # 计算百分位数
                percentile = self.calculate_percentile(series, value)

                indicators[indicator_name] = {
                    'name': name_mapping.get(indicator_name, indicator_name),
                    'raw_value': float(value),
                    'unit': units.get(indicator_name, ''),
                    'percentile_5y': percentile,
                    'interpretation': self.interpret_percentile(percentile),
                    'frequency': series.attrs.get('freq', 'unknown')
                }

        return indicators

    def calculate_price_path(self, event_date: str,
                             price_type: str = 'brent',
                             lookback_days: int = 60,
                             lookforward_days: int = 60) -> Dict[str, Any]:
        """
        计算价格路径（增强版）
        """
        if self.price_df is None or price_type not in self.price_df.columns:
            return {}

        event_dt = pd.to_datetime(event_date)
        price_series = self.price_df[price_type].dropna()

        if len(price_series) == 0:
            return {}

        # 获取事件前的价格（最近的有效价格）
        pre_prices = price_series[price_series.index < event_dt]
        if len(pre_prices) == 0:
            return {}

        pre_date = pre_prices.index[-1]
        pre_price = pre_prices.iloc[-1]

        price_path = {
            'pre': {
                'date': pre_date.strftime('%Y-%m-%d'),
                'close': float(pre_price)
            },
            'daily': [],
            'peak': {'value': None, 'day': None, 'date': None, 'return_since_pre': None},
            'trough': {'value': None, 'day': None, 'date': None, 'return_since_pre': None},
            'volatility_pre': None,
            'volatility_post': None,
            'pattern': '',
            'pattern_interpretation': ''
        }

        # 定义需要记录的时间点（交易日）
        time_points = {
            't0': 0,
            't+1': 1,
            't+2': 2,
            't+3': 3,
            't+4': 4,
            't+5': 5,
            't+10': 10,
            't+20': 20,
            't+30': 30,
            't+60': 60
        }

        # 获取事件后的价格序列
        post_prices = price_series[price_series.index >= event_dt]

        # 计算每个时间点的价格
        for day_name, days_offset in time_points.items():
            if days_offset == 0:
                # t0：事件发生后第一个有效价格
                if len(post_prices) > 0:
                    date = post_prices.index[0]
                    close = post_prices.iloc[0]
                else:
                    continue
            else:
                # 查找距离事件日 days_offset 个交易日的价格
                if len(post_prices) > days_offset:
                    date = post_prices.index[days_offset]
                    close = post_prices.iloc[days_offset]
                else:
                    continue

            return_pct = (close - pre_price) / pre_price * 100
            price_path['daily'].append({
                'day': day_name,
                'date': date.strftime('%Y-%m-%d'),
                'close': float(close),
                'return_pct': round(float(return_pct), 2)
            })

        # 计算峰值和谷值（事件后lookforward_days天内）
        post_period_prices = post_prices[
            (post_prices.index >= event_dt) &  # 改为 >=，不包括事件前
            (post_prices.index <= event_dt + timedelta(days=lookforward_days))
            ]

        if len(post_period_prices) > 0 and pre_price:
            # 峰值
            peak_idx = post_period_prices.idxmax()
            peak_value = post_period_prices.max()
            days_since_event = (peak_idx - event_dt).days
            price_path['peak'] = {
                'value': float(peak_value),
                'day': days_since_event,
                'date': peak_idx.strftime('%Y-%m-%d'),
                'return_since_pre': round((peak_value - pre_price) / pre_price * 100, 2)
            }

            # 谷值
            trough_idx = post_period_prices.idxmin()
            trough_value = post_period_prices.min()
            days_since_event = (trough_idx - event_dt).days
            price_path['trough'] = {
                'value': float(trough_value),
                'day': days_since_event,
                'date': trough_idx.strftime('%Y-%m-%d'),
                'return_since_pre': round((trough_value - pre_price) / pre_price * 100, 2)
            }

            # 计算波动率（年化）
            # 事件前波动率
            pre_period_prices = price_series[
                (price_series.index >= event_dt - timedelta(days=lookback_days)) &
                (price_series.index < event_dt)
                ]
            if len(pre_period_prices) > 1:
                pre_returns = pre_period_prices.pct_change().dropna()
                if len(pre_returns) > 0:
                    price_path['volatility_pre'] = round(pre_returns.std() * np.sqrt(252) * 100, 2)

            # 事件后波动率
            if len(post_period_prices) > 1:
                post_returns = post_period_prices.pct_change().dropna()
                if len(post_returns) > 0:
                    price_path['volatility_post'] = round(post_returns.std() * np.sqrt(252) * 100, 2)

        return price_path

    def fill_with_llm(self, event_info, template, price_path):
        if not self.api_key:
            print("未提供API密钥，跳过LLM填充")
            return template

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

        # 构建提示词（添加 JSON 格式强调）
        prompt = f"""
        请根据以下历史事件信息，填充历史事件数据库模板中的字段。

        事件信息：
        - 事件ID：{event_info.get('event_id', '')}
        - 事件名称：{event_info.get('event_name', '')}
        - 事件日期：{event_info.get('event_date', '')}
        - 事件地点：{event_info.get('location', '')}
        - 事件类型：{event_info.get('event_type', '')}
        - 事件子类型：{event_info.get('subtype', '')}
        - 涉及国家：{', '.join(event_info.get('involved_countries', []))}
        - 简要描述：{event_info.get('brief_description', '')}
        """

        if price_path and price_path.get('daily'):
            prompt += f"""
        价格路径信息：
        - 事件前价格：{price_path.get('pre', {}).get('close', 'N/A')} 美元/桶
        - 峰值：{price_path.get('peak', {}).get('value', 'N/A')} 美元/桶
        - 谷值：{price_path.get('trough', {}).get('value', 'N/A')} 美元/桶
        """

        prompt += """

        请返回纯JSON格式（不要包含任何其他文字），字段如下：
        {
            "pre_event_timeline": [{"date": "YYYY-MM-DD", "event": "描述"}],
            "post_event_timeline": [{"date": "YYYY-MM-DD", "event": "描述"}],
            "turning_points": [{"day": 数字, "date": "YYYY-MM-DD", "event": "描述", "impact": "上涨/下跌/反转", "magnitude": "幅度"}],
            "summary": "一句话总结",
            "price_pattern": "冲高回落/持续上涨/快速修复/震荡",
            "pattern_interpretation": "价格走势特征描述"
        }
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个历史事件分析专家。必须只返回JSON格式，不要添加任何解释文字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            # 获取响应内容
            content = response.choices[0].message.content.strip()
            print(f"LLM返回内容: {content[:200]}...")  # 调试输出

            # 尝试提取JSON（处理可能被markdown包裹的情况）
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()

            llm_output = json.loads(content)

            # 更新模板
            template['pre_event_timeline'] = llm_output.get('pre_event_timeline', [])
            template['post_event_timeline'] = llm_output.get('post_event_timeline', [])
            template['turning_points'] = llm_output.get('turning_points', [])
            template['summary'] = llm_output.get('summary', '')

            if 'price_path' in template:
                for price_type in ['brent', 'wti']:
                    if price_type in template['price_path']:
                        template['price_path'][price_type]['pattern'] = llm_output.get('price_pattern', '')
                        template['price_path'][price_type]['pattern_interpretation'] = llm_output.get(
                            'pattern_interpretation', '')

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始返回: {content}")
        except Exception as e:
            print(f"LLM填充失败: {e}")
            import traceback
            traceback.print_exc()

        return template

    def auto_generate_event(self,
                            event_name: str,
                            event_date: str = None,
                            use_llm: bool = True) -> Dict[str, Any]:
        """
        仅通过事件名称自动生成完整事件条目

        Parameters:
        -----------
        event_name : str
            事件名称（如"俄乌冲突爆发"）
        event_date : str
            事件日期（可选，如果不提供LLM会尝试推断）
        use_llm : bool
            是否使用LLM
        """

        if not self.api_key:
            print("需要API密钥才能自动生成事件信息")
            return {}

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

        # 构建提示词，让LLM生成完整的事件信息
        prompt = f"""
        请根据事件名称 "{event_name}"，生成完整的历史事件信息。

        {f"事件日期参考：{event_date}" if event_date else "请推断事件发生的具体日期"}

        【重要要求】
        1. 每个时间线事件必须包含可信来源
        2. 来源可以是：新闻媒体名称（如"路透社"、"新华社"）、官方声明、权威机构报告
        3. 如果知道具体URL可以给出，不知道则给出来源名称即可
        4. 来源要真实可查，不要编造

        请返回纯JSON格式，包含以下字段：

        {{
          "event_id": "YYYYMMDD_事件缩写",
          "event_date": "YYYY-MM-DD",
          "location": "事件发生地点",
          "event_type": "地缘冲突/宏观政策/供给冲击/需求冲击",
          "subtype": "战争/制裁/加息/减产/疫情等",
          "involved_countries": ["国家1", "国家2"],
          "brief_description": "一句话描述事件核心（50字以内）",

          "pre_event_timeline": [
            {{
              "date": "YYYY-MM-DD",
              "event": "事件描述",
              "source": "来源名称或URL",
              "source_description": "来源简要说明"
            }}
          ],

          "post_event_timeline": [
            {{
              "date": "YYYY-MM-DD",
              "event": "事件描述",
              "source": "来源名称或URL",
              "source_description": "来源简要说明"
            }}
          ],

          "turning_points": [
            {{
              "day": 数字,
              "date": "YYYY-MM-DD",
              "event": "转折事件描述",
              "impact": "上涨/下跌/反转",
              "magnitude": "影响幅度",
              "source": "来源名称或URL"
            }}
          ],

          "summary": "一句话总结（100字以内）",
          "price_pattern": "冲高回落/持续上涨/快速修复/震荡",
          "pattern_interpretation": "价格走势特征描述"
        }}

        要求：
        - 时间线至少3个前序事件、3个后续事件
        - 转折点至少2个
        - 来源必须真实可信，不要编造
        - 只返回JSON，不要其他内容
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "你是一个历史事件分析专家，精通全球经济、政治和能源市场。必须只返回JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # 清理markdown标记
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            event_info = json.loads(content)

            # 使用获取的信息构建事件
            event_entry = self.build_event_entry(
                event_id=event_info.get('event_id',
                                        f"{event_info.get('event_date', '').replace('-', '')}_{event_name[:10]}"),
                event_name=event_name,
                event_date=event_info.get('event_date', event_date if event_date else "2020-01-01"),
                event_type=event_info.get('event_type', '宏观政策'),
                subtype=event_info.get('subtype', '其他'),
                location=event_info.get('location', ''),
                involved_countries=event_info.get('involved_countries', []),
                brief_description=event_info.get('brief_description', ''),
                use_llm=False  # 避免递归调用
            )

            # 补充LLM生成的时间线和转折点
            event_entry['pre_event_timeline'] = event_info.get('pre_event_timeline', [])
            event_entry['post_event_timeline'] = event_info.get('post_event_timeline', [])
            event_entry['turning_points'] = event_info.get('turning_points', [])
            event_entry['summary'] = event_info.get('summary', '')

            if 'price_path' in event_entry:
                for price_type in ['brent', 'wti']:
                    if price_type in event_entry['price_path']:
                        event_entry['price_path'][price_type]['pattern'] = event_info.get('price_pattern', '')
                        event_entry['price_path'][price_type]['pattern_interpretation'] = event_info.get(
                            'pattern_interpretation', '')

            return event_entry

        except Exception as e:
            print(f"自动生成事件失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def build_event_entry(self,
                          event_id: str,
                          event_name: str,
                          event_date: str,
                          event_type: str,
                          subtype: str,
                          location: str = "",
                          involved_countries: List[str] = None,
                          brief_description: str = "",
                          use_llm: bool = True,
                          price_types: List[str] = ['brent', 'wti']) -> Dict[str, Any]:
        """
        构建单个事件条目（增强版）
        """

        # 初始化模板
        template = {
            "event_id": event_id,
            "event_name": event_name,
            "event_date": event_date,
            "event_timezone": "UTC+8",
            "location": location,
            "event_type": event_type,
            "subtype": subtype,
            "involved_countries": involved_countries or [],
            "brief_description": brief_description,
            "pre_event_timeline": [],
            "post_event_timeline": [],
            "core_indicators": {},
            "price_path": {},
            "turning_points": [],
            "summary": "",
            "data_sources": {
                "indicators": "CSV files",
                "price": "CSV files",
                "timeline": "LLM生成+人工复核" if use_llm else "待填写",
                "verified_by": "",
                "verified_date": ""
            }
        }

        # 自动填充指标数据
        indicators = self.get_indicators_at_date(event_date)
        template['core_indicators'] = indicators

        # 自动填充价格路径
        price_path_result = {}
        if self.price_df is not None:
            for price_type in price_types:
                if price_type in self.price_df.columns:
                    price_path = self.calculate_price_path(event_date, price_type)
                    if price_path:
                        price_path_result[price_type] = price_path

        template['price_path'] = price_path_result

        # 使用LLM填充其他字段
        if use_llm:
            event_info = {
                'event_id': event_id,
                'event_name': event_name,
                'event_date': event_date,
                'location': location,
                'event_type': event_type,
                'subtype': subtype,
                'involved_countries': involved_countries,
                'brief_description': brief_description
            }
            # 使用布伦特价格路径作为参考
            brent_path = price_path_result.get('brent', {})
            template = self.fill_with_llm(event_info, template, brent_path)

        return template

    def batch_build_events(self, events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量构建事件条目"""
        results = []
        for i, event in enumerate(events_list):
            print(f"处理事件 {i + 1}/{len(events_list)}: {event.get('event_name', '')}")
            entry = self.build_event_entry(**event)
            results.append(entry)
        return results

    def save_to_json(self, entries: List[Dict[str, Any]], output_file: str):
        """保存事件条目到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(entries)} 个事件到 {output_file}")

    def export_to_excel(self, entries: List[Dict[str, Any]], output_file: str):
        """导出到Excel文件（便于人工复核）"""
        # 展平数据结构以便导出
        flat_data = []
        for entry in entries:
            flat_entry = {
                'event_id': entry['event_id'],
                'event_name': entry['event_name'],
                'event_date': entry['event_date'],
                'location': entry['location'],
                'event_type': entry['event_type'],
                'subtype': entry['subtype'],
                'involved_countries': ', '.join(entry['involved_countries']),
                'brief_description': entry['brief_description'],
                'summary': entry['summary']
            }

            # 添加指标数据
            for indicator, data in entry['core_indicators'].items():
                flat_entry[f'{indicator}_value'] = data.get('raw_value', '')
                flat_entry[f'{indicator}_percentile'] = data.get('percentile_5y', '')
                flat_entry[f'{indicator}_interpretation'] = data.get('interpretation', '')

            # 添加价格数据
            if 'brent' in entry['price_path']:
                brent = entry['price_path']['brent']
                flat_entry['brent_pre_price'] = brent.get('pre', {}).get('close', '')
                flat_entry['brent_peak'] = brent.get('peak', {}).get('value', '')
                flat_entry['brent_trough'] = brent.get('trough', {}).get('value', '')
                flat_entry['brent_pattern'] = brent.get('pattern', '')

            flat_data.append(flat_entry)

        df = pd.DataFrame(flat_data)
        df.to_excel(output_file, index=False)
        print(f"已导出到Excel: {output_file}")


# ==================== 使用示例 ====================

def main():
    # 1. 设置数据文件夹路径
    data_folder = "data"  # 存放所有CSV文件的文件夹

    # 2. 创建配置（可选，如果不提供则自动扫描）
    indicator_config = {
        'inventory': {
            'file': 'us_crude_inventory.csv',
            'date_col': 'date',
            'value_col': 'value',
            'freq': 'W'  # 周度数据
        },
        'opec_spare': {
            'file': 'opec.csv',
            'date_col': 'date',
            'value_col': 'capacity',
            'freq': 'M'  # 月度数据
        },
        'brent_price': {
            'file': 'brent_price.csv',
            'date_col': 'date',
            'value_col': 'close',
            'freq': 'D'  # 日度数据
        },
        'dxy': {
            'file': 'dxy.csv',
            'date_col': 'date',
            'value_col': 'close',
            'freq': 'D'
        },
        'vix': {
            'file': 'vix.csv',
            'date_col': 'date',
            'value_col': 'close',
            'freq': 'D'
        },
        'refinery_util': {
            'file': 'refinery_utilization.csv',
            'date_col': 'date',
            'value_col': 'rate',
            'freq': 'M'
        }
    }

    # 3. 初始化构建器
    builder = HistoricalEventDatabaseBuilder(
        data_folder=data_folder,
        api_key=os.getenv('OPENAI_API_KEY')
    )

    # 4. 加载价格数据（如果单独存储）
    builder.load_price_data(
        brent_file=str(Path(data_folder) / "brent_price.csv"),
        date_column='date',
        value_column='value'  # 指定为 'value'
    )

    # 事件清单
    # events = [
    #     "海湾战争爆发",
    #     "伊拉克战争爆发",
    #     "利比亚内战爆发",
    #     "伊朗核设施遭袭",
    #     "也门胡塞武装袭击沙特油田",
    #     "阿联酋油轮遭袭",
    #     "以色列空袭伊朗石油设施",
    #     "尼日利亚三角洲武装袭击",
    #     "OPEC+达成历史性减产协议",
    #     "OPEC+增产谈判破裂",
    #     "OPEC+减产协议到期",
    #     "沙特宣布自愿额外减产",
    #     "OPEC+会议推迟",
    #     "沙特俄罗斯石油价格战",
    #     "美国重启对伊朗石油制裁",
    #     "美国宣布对委内瑞拉石油制裁",
    #     "欧盟宣布禁运俄罗斯石油",
    #     "G7宣布对俄罗斯石油限价",
    #     "美国释放战略石油储备",
    #     "新冠疫情全球大流行",
    #     "全球石油需求创纪录暴跌",
    #     "美国页岩油破产潮",
    #     "苏伊士运河堵塞",
    #     "沙特石油设施遭无人机袭击",
    #     "美联储开启加息周期",
    #     "美联储紧急降息",
    #     "美元指数突破新高",
    #     "全球金融危机爆发"
    # ]

    # 创建输出文件夹
    output_dir = Path("output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    event_name = "俄乌冲突爆发"
    event = builder.auto_generate_event(event_name)
    if event and event.get('event_id'):
        # 生成文件名（使用 event_id）
        filename = f"{event['event_id']}.json"
        filepath = output_dir / filename

        # 单独保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(event, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 已保存: {filename}")

    # # 批量构建并单独保存
    # success_count = 0
    # fail_count = 0
    #
    # for event_name in events:
    #     print(f"处理: {event_name}")
    #
    #     try:
    #         # 自动生成事件
    #         event = builder.auto_generate_event(event_name)
    #
    #         if event and event.get('event_id'):
    #             # 生成文件名（使用 event_id）
    #             filename = f"{event['event_id']}.json"
    #             filepath = output_dir / filename
    #
    #             # 单独保存
    #             with open(filepath, 'w', encoding='utf-8') as f:
    #                 json.dump(event, f, ensure_ascii=False, indent=2)
    #
    #             print(f"  ✅ 已保存: {filename}")
    #             success_count += 1
    #         else:
    #             print(f"  ❌ 生成失败")
    #             fail_count += 1
    #
    #     except Exception as e:
    #         print(f"  ❌ 异常: {e}")
    #         fail_count += 1
    #
    # print(f"\n完成！成功: {success_count}, 失败: {fail_count}")
    # print(f"文件保存在: {output_dir}")


if __name__ == "__main__":
    main()