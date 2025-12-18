import pandas as pd

# 读取parquet文件
df = pd.read_parquet('./stage2_test.parquet')

def query_by_img_file(img_file):
    # 查找person字段以/{img_file}结尾的行
    result = df[df['person'].str.endswith(f'/{img_file}')]
    if result.empty:
        print(f"未找到 img_file = {img_file} 的条目")
    else:
        # 如果有多条，全部输出
        for record in result.to_dict(orient='records'):
            print(record)

if __name__ == "__main__":
    img_file = input("请输入img_file（如 123456）: ").strip()
    query_by_img_file(img_file)