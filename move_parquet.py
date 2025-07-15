import pandas as pd

# 设置移动的条数n
n = 5  # 可根据需要修改

# 读取原始parquet文件
df = pd.read_parquet('./stage2_test.parquet')

# 1. 先提取最后n条
def get_last_n(series, n):
    return series.iloc[-n:].values

last_n_bg_depth = get_last_n(df['bg_depth'], n)
last_n_bg = get_last_n(df['bg'], n)
last_n_lighting = get_last_n(df['lighting'], n)

# 2. 整体向下移动n个条目
bg_depth_shifted = df['bg_depth'].shift(n)
bg_shifted = df['bg'].shift(n)
lighting_shifted = df['lighting'].shift(n)

# 3. 用last_n填充最前面的n个条目
bg_depth_shifted.iloc[:n] = last_n_bg_depth
bg_shifted.iloc[:n] = last_n_bg
lighting_shifted.iloc[:n] = last_n_lighting

# 4. 更新DataFrame并保存
new_df = df.copy()
new_df['bg_depth'] = bg_depth_shifted
new_df['bg'] = bg_shifted
new_df['lighting'] = lighting_shifted
new_df.to_parquet('./stage2_inthewild_lighting.parquet')

print(f'向下移动{n}条，前{n}条用原本最后{n}条填充，保存为stage2_inthewild_lighting.parquet') 
