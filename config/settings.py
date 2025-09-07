"""
全局配置文件
包含项目的所有配置参数，用于统一管理项目设置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NETWORK_DATA_DIR = DATA_DIR / "network"

# 原始数据子目录
LANGCHAIN_RAW_DATA_DIR = RAW_DATA_DIR / "langchain_ecosystem_data"
ETL_OUTPUT_DIR = PROCESSED_DATA_DIR / "etl_output"
RELEVANCE_OUTPUT_DIR = PROCESSED_DATA_DIR / "relevance_judgment_output"
CLASSIFICATION_OUTPUT_DIR = PROCESSED_DATA_DIR / "classification_output"
FINAL_ANALYSIS_DATA_DIR = PROCESSED_DATA_DIR / "final_analysis_data"
NETWORK_OUTPUT_DIR = NETWORK_DATA_DIR / "network_output"

# 结果目录配置
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_OUTPUT_DIR = RESULTS_DIR / "analysis_output"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

# 分析参数配置
ANALYSIS_CONFIG = {
    # 时间范围设置
    "start_date": "2022-11-01",
    "end_date": "2024-12-31",
    
    # 网络分析参数
    "centrality_types": ["degree", "betweenness", "eigenvector", "closeness"],
    "community_algorithms": ["louvain", "leiden", "infomap"],
    
    # 用户角色聚类参数
    "clustering_algorithms": ["kmeans", "gmm", "hierarchical"],
    "k_range": range(2, 16),
    
    # LangChain核心团队成员
    "core_team_logins": [
        # 创始人
        "hwchase17",  # Harrison Chase
        "agola11",    # Ankush Gola
        # 核心维护者
        "baskaryan", "ccurme", "hinthornw", "rlancemartin",
        "nfcampos", "vbarda", "efriis", "eyurtsev"
    ]
}

# 模型配置
MODEL_CONFIG = {
    # HAN模型参数
    "han_model": {
        "num_heads": 8,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": 100
    },
    
    # 聚类模型参数
    "clustering": {
        "kmeans_params": {
            "random_state": 42,
            "n_init": 10,
            "max_iter": 300
        },
        "gmm_params": {
            "random_state": 42,
            "max_iter": 100,
            "covariance_type": "full"
        }
    }
}

# 模型配置
MODEL_CONFIG = {
    "save_models": True,
    "model_checkpoint_dir": MODELS_DIR,
    "han_model": {
        "hidden_dim": 128,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "epochs": 100,
        "learning_rate": 0.001,
        "patience": 10
    },
    "node2vec": {
        "embedding_dim": 128,
        "walk_length": 20,
        "context_size": 10,
        "walks_per_node": 10,
        "p": 1.0,
        "q": 1.0,
        "epochs": 100
    },
    "graphsage": {
        "hidden_dim": 64,
        "output_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "epochs": 200
    }
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "figure_size": (15, 8),
    "dpi": 300,
    "style": "seaborn-v0_8-whitegrid",
    "color_palette": "viridis",
    "font_size": 12,
    "title_font_size": 16
}

# 文件名常量
FILENAMES = {
    "graph_file": "full_ecosystem_graph.graphml",
    "panel_data": "monthly_panel_data.csv",
    "user_features": "user_behavior_features.csv",
    "user_roles": "user_roles.csv",
    "community_details": "community_details_by_month.csv",
    "repo_classification": "repos_classified.csv"
}

# 确保所有目录存在
def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, NETWORK_DATA_DIR,
        LANGCHAIN_RAW_DATA_DIR, ETL_OUTPUT_DIR, RELEVANCE_OUTPUT_DIR,
        CLASSIFICATION_OUTPUT_DIR, FINAL_ANALYSIS_DATA_DIR, NETWORK_OUTPUT_DIR,
        RESULTS_DIR, ANALYSIS_OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 在导入时确保目录存在
ensure_directories()
