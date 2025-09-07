import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
ANALYSIS_OUTPUT_DIR = "analysis_output"
FEATURES_FILENAME = "user_behavior_features.csv"
ROLES_FILENAME = "user_roles_final.csv"  # 更新输出文件名


# ... 其他文件名 ...

# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    print("--- RQ2 分析 (V2): 两阶段用户角色聚类 ---")

    # --- 1. 加载和准备数据 ---
    features_path = os.path.join(ANALYSIS_OUTPUT_DIR, FEATURES_FILENAME)
    features_df = pd.read_csv(features_path)
    feature_cols = [col for col in features_df.columns if col not in ['user_id', 'login']]
    X = features_df[feature_cols]

    print(f"[*] 成功加载 {len(features_df)} 位用户的行为特征。")

    # --- 2. 第一阶段聚类: 分离“沉默的大多数”与“活跃核心” ---
    print("\n--- 阶段一: 分离 沉默的大多数 ---")
    scaler1 = StandardScaler()
    X_scaled1 = scaler1.fit_transform(X)

    # 根据之前的肘部图，我们已知K=6可以很好地分离出异常点，这里直接使用
    # 或者可以使用简单的K=2来做“活跃”与“不活跃”的二分
    kmeans1 = KMeans(n_clusters=6, random_state=42, n_init=10)
    features_df['pass1_cluster'] = kmeans1.fit_predict(X_scaled1)

    # 找到那个最大的、代表“沉默的大多数”的聚类ID
    majority_cluster_id = features_df['pass1_cluster'].mode()[0]
    population = features_df['pass1_cluster'].value_counts()
    print(
        f"[✓] 第一阶段聚类完成。识别出“沉默的大多数”的聚类ID为: {majority_cluster_id} (人数: {population[majority_cluster_id]})")

    # --- 3. 准备第二阶段数据：筛选出“活跃核心”用户 ---
    active_users_df = features_df[features_df['pass1_cluster'] != majority_cluster_id].copy()
    print(f"[*] 已筛选出 {len(active_users_df)} 位活跃核心用户，准备进行第二阶段聚类。")

    # 准备用于第二阶段聚类的数据
    X_active = active_users_df[feature_cols]
    scaler2 = StandardScaler()
    X_active_scaled = scaler2.fit_transform(X_active)

    # --- 4. 第二阶段: 对“活跃核心”进行深度聚类 ---
    print("\n--- 阶段二: 对活跃核心用户进行深度角色挖掘 ---")
    print("[*] 正在为活跃核心用户重新运行“肘部法则”...")
    inertia = []
    k_range = range(2, 11)  # 对活跃用户，我们通常期望的角色数量不会太多
    for k in tqdm(k_range, desc="测试活跃用户的K值"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_active_scaled)
        inertia.append(kmeans.inertia_)

    # 绘制新的肘部图
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Active Roles (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Active Core Users')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "kmeans_elbow_plot_active_users.png"), dpi=300)
    print("[!] 请查看为“活跃核心用户”新生成的肘部图，并确定新的K值。")
    plt.show()

    # !!! 行动要求: 请根据新图，在此处填入您为活跃用户选择的最佳K值 !!!
    CHOSEN_K_ACTIVE = 5
    print(f"\n[*] 已选择 K={CHOSEN_K_ACTIVE}。开始对活跃用户进行最终聚类...")

    kmeans2 = KMeans(n_clusters=CHOSEN_K_ACTIVE, random_state=42, n_init=10)
    active_users_df['active_role_cluster'] = kmeans2.fit_predict(X_active_scaled)

    # --- 5. 角色画像与合并 ---
    print("\n--- 活跃核心用户的角色画像 ---")
    active_profile_df = active_users_df.groupby('active_role_cluster')[feature_cols].mean()
    active_profile_df['population'] = active_users_df['active_role_cluster'].value_counts()
    print(active_profile_df)

    # 可视化
    plt.figure(figsize=(16, 8))
    profile_scaled = scaler2.fit_transform(active_profile_df[feature_cols])
    profile_scaled_df = pd.DataFrame(profile_scaled, index=active_profile_df.index, columns=feature_cols)
    sns.heatmap(profile_scaled_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Heatmap of Active User Role Profiles')
    plt.xlabel('Behavioral Features')
    plt.ylabel('Active Role (Cluster ID)')
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "user_role_profiles_heatmap_active.png"), dpi=300)
    plt.show()

    # --- 6. 生成最终的角色定义文件 ---
    # a. 为所有用户创建一个最终的角色列
    final_roles = {}
    # 为沉默的大多数命名
    final_roles[majority_cluster_id] = "Casual Observer"

    # 为活跃角色命名 (您可以根据画像结果修改这些名称)
    # 这是一个示例，您需要根据上一步打印的画像表格来手动命名
    role_names = [f"Active Role {i}" for i in range(CHOSEN_K_ACTIVE)]  # 临时名称

    # 将活跃用户的聚类结果映射到总的用户表中
    features_df['final_role'] = features_df['pass1_cluster'].apply(lambda x: final_roles.get(x))

    # 创建一个从活跃聚类ID -> 最终角色名的映射
    active_role_map = {i: f"Active Role {i}" for i in range(CHOSEN_K_ACTIVE)}  # 您需要根据画像结果修改这里的名称

    # 合并活跃用户的角色
    final_role_series = active_users_df['active_role_cluster'].map(active_role_map)
    features_df.loc[final_role_series.index, 'final_role'] = final_role_series

    # 选择最终需要的列
    final_df = features_df[['user_id', 'login', 'final_role'] + feature_cols]

    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, ROLES_FILENAME)
    final_df.to_csv(output_path, index=False)

    print(f"\n--- 两阶段用户角色识别完成 ---")
    print(f"[*] 最终的用户角色数据已保存至: '{output_path}'")
    print("\n--- 最终角色分布情况 ---")
    print(final_df['final_role'].value_counts())


if __name__ == "__main__":
    main()