import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np  # 导入NumPy库用于对数变换

# ==============================================================================
# --- 1. 配置区域 (保持不变) ---
# ==============================================================================
ANALYSIS_OUTPUT_DIR = "analysis_output"
FEATURES_FILENAME = "user_behavior_features.csv"
ROLES_FILENAME = "user_roles_final.csv"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    print("--- RQ2 分析 (V3): 两阶段聚类 + 对数变换 ---")

    # --- 1. 加载数据 (与V2相同) ---
    features_path = os.path.join(ANALYSIS_OUTPUT_DIR, FEATURES_FILENAME)
    features_df = pd.read_csv(features_path)
    feature_cols = [col for col in features_df.columns if col not in ['user_id', 'login']]
    X = features_df[feature_cols]

    # --- 2. 第一阶段聚类 (与V2相同) ---
    print("\n--- 阶段一: 分离 沉默的大多数 ---")
    scaler1 = StandardScaler()
    X_scaled1 = scaler1.fit_transform(X)
    kmeans1 = KMeans(n_clusters=6, random_state=42, n_init=10)
    features_df['pass1_cluster'] = kmeans1.fit_predict(X_scaled1)
    majority_cluster_id = features_df['pass1_cluster'].mode()[0]
    active_users_df = features_df[features_df['pass1_cluster'] != majority_cluster_id].copy()
    print(f"[*] 已筛选出 {len(active_users_df)} 位活跃核心用户。")

    # --- 3. 第二阶段: 对“活跃核心”进行深度聚类 ---
    print("\n--- 阶段二: 对活跃核心用户进行深度角色挖掘 ---")
    X_active_original = active_users_df[feature_cols]

    # --- 关键步骤：对特征进行对数变换 (Log Transform) ---
    # np.log1p(x) 计算 log(1+x)，可以优雅地处理0值
    print("[*] 正在对活跃用户的行为特征进行对数变换以减小极值影响...")
    X_active_log = np.log1p(X_active_original)

    # 在对数变换后的数据上进行标准化
    scaler2 = StandardScaler()
    X_active_scaled = scaler2.fit_transform(X_active_log)
    print("[✓] 对数变换与标准化完成。")

    # 重新运行“肘部法则”
    print("[*] 正在为变换后的数据重新运行“肘部法则”...")
    inertia = []
    k_range = range(2, 11)
    for k in tqdm(k_range, desc="测试K值"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_active_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Active Roles (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Log-Transformed Active Users')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "kmeans_elbow_plot_log_active.png"), dpi=300)
    print("[!] 请查看为“对数变换后”数据新生成的肘部图，并确定新的K值。")
    plt.show()

    # !!! 行动要求: 请根据新图，在此处填入您选择的最佳K值 !!!
    CHOSEN_K_ACTIVE = 5
    print(f"\n[*] 已选择 K={CHOSEN_K_ACTIVE}。开始对变换后的数据进行最终聚类...")

    kmeans2 = KMeans(n_clusters=CHOSEN_K_ACTIVE, random_state=42, n_init=10)
    active_users_df['active_role_cluster'] = kmeans2.fit_predict(X_active_scaled)

    # --- 4. 角色画像与合并 ---
    print("\n--- 活跃核心用户的角色画像 (基于原始行为均值) ---")
    # 注意：画像时，我们使用原始的、未经变换的数据进行 .mean() 计算，因为这样结果才具有解释性
    active_profile_df = active_users_df.groupby('active_role_cluster')[feature_cols].mean()
    active_profile_df['population'] = active_users_df['active_role_cluster'].value_counts()
    print(active_profile_df)

    # 可视化
    profile_scaled = scaler2.transform(np.log1p(active_profile_df[feature_cols]))
    profile_scaled_df = pd.DataFrame(profile_scaled, index=active_profile_df.index, columns=feature_cols)
    plt.figure(figsize=(16, 8))
    sns.heatmap(profile_scaled_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Heatmap of Active User Role Profiles (Log-Transformed & Scaled)')
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "user_role_profiles_heatmap_log_active.png"), dpi=300)
    plt.show()

    # --- 5. 生成最终的角色定义文件 (手动命名) ---
    print("\n[*] 正在生成最终角色定义文件...")
    # 在这里，您可以根据上面打印的画像表格，为每个角色赋予有意义的名称

    role_name_map = {
        0: "社区枢纽 (Community Hub)",
        1: "高产开发者 (Prolific Developer)",
        2: "核心贡献者 (Core Contributor)",
        3: "生态猎手 (Ecosystem Scout)",
        4: "基础用户 (Regular User)"
    }

    # 创建最终的角色列
    # 1. 默认所有用户都是轻度关注者
    features_df['final_role'] = "Casual Observer"
    # 2. 获取活跃用户的聚类结果
    active_clusters = active_users_df['active_role_cluster']
    # 3. 将活跃用户的角色名填入总表
    features_df.loc[active_clusters.index, 'final_role'] = active_clusters.map(role_name_map).fillna(
        "Active User (unnamed)")

    final_df = features_df[['user_id', 'login', 'final_role'] + feature_cols]
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, ROLES_FILENAME)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n--- V3 用户角色识别完成 ---")
    print(f"[*] 最终的用户角色数据已保存至: '{output_path}'")
    print("\n--- 最终角色分布情况 ---")
    print(final_df['final_role'].value_counts())


if __name__ == "__main__":
    main()