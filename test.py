import pandas as pd

# 查看你的月度数据
df = pd.read_csv("data/opec.csv")
print(df[df['date'].str.contains('2022-02')])  # 看看2月的具体日期