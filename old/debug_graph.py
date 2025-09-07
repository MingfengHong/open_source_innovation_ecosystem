import networkx as nx
import pandas as pd

# 加载图
print("[*] 正在加载图文件...")
G = nx.read_graphml("network_output/full_ecosystem_graph.graphml")
print("[✓] 图加载完毕。")

# 检查前10条带时间戳的边
print("\n--- 检查图中边的'timestamp'属性 ---")
count = 0
for u, v, data in G.edges(data=True):
    if 'timestamp' in data:
        print(f"Edge ({u}, {v}) -> timestamp: {data['timestamp']}")
        count += 1
        if count >= 10:
            break

# 隔离测试时间戳比较逻辑
print("\n--- 隔离测试时间戳比较 ---")
# 从上面输出中随便选一个时间戳字符串
sample_ts_str = "2023-10-15T20:30:45Z" # 假设这是我们找到的一个时间戳

# 模拟主脚本中的逻辑
ts_from_graph = pd.to_datetime(sample_ts_str)
start_of_month_naive = pd.to_datetime("2023-10-01")

print(f"从图中解析的时间戳 (ts_from_graph): {ts_from_graph}")
print(f"脚本中创建的月份开始时间 (start_of_month_naive): {start_of_month_naive}")

# 关键：移除时区信息，使其可以和naive时间进行比较
ts_naive = ts_from_graph.tz_localize(None)
print(f"移除时区信息后的时间戳 (ts_naive): {ts_naive}")

# 现在进行比较
is_in_range = start_of_month_naive <= ts_naive
print(f"\n比较结果 (start <= ts_naive): {is_in_range}")