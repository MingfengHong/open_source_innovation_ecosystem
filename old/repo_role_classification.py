import os
import json
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# --- 1. 配置区域 & 关键词集合 ---
# ==============================================================================
# 输入目录: 包含从数据收集中获取的原始JSON文件的文件夹
INPUT_DIR = "langchain_ecosystem_data"
# 输出目录: 用于存放分类后的CSV文件
OUTPUT_DIR = "classification_output"

# 用于描述和主题分析的关键词集合
# 这些关键词可以在后续的验证环节中进行优化和调整
APP_KWS = {'app', 'ui', 'demo', 'interface', 'chatbot', 'agent', 'service', 'platform', 'app-builder', '应用', '前端',
           '聊天机器人'}
TOOL_KWS = {'tool', 'monitor', 'monitoring', 'observability', 'debug', 'debugging', 'eval', 'evaluation', 'deploy',
            'deployment', 'testing', 'trace', 'tracing', '工具', '监控', '评估', '部署', '测试'}
LIB_KWS = {'library', 'plugin', 'integration', 'connector', 'wrapper', 'sdk', 'package', '库', '插件', '集成', '连接器'}
EDU_KWS = {'tutorial', 'course', 'notebooks', 'examples', 'awesome', 'paper', 'reproducibility', 'learn', '教程',
           '课程', '案例', '论文', '学习'}
BOILERPLATE_KWS = {'template', 'starter', 'boilerplate', 'cookiecutter', '模板', '脚手架'}


# ==============================================================================
# --- 2. 特征工程 (构建信号矩阵) ---
# ==============================================================================

def extract_signals(repo_data):
    """
    从单个仓库的JSON数据中提取一个特征向量（一个包含所有信号的字典）。
    """
    if not repo_data:
        return None

    signals = {}  # 初始化信号字典
    name = repo_data.get('nameWithOwner', '').lower()  # 获取仓库全名并转为小写
    description = (repo_data.get('description') or '').lower()  # 获取描述并转为小写
    topics_data = repo_data.get('repositoryTopics', {}) or {}  # 获取主题数据
    # 将所有主题转为小写，并存入一个集合
    topics = {t['topic']['name'].lower() for t in topics_data.get('nodes', []) if t and t.get('topic')}

    # 从PR中获取文件路径列表，以此作为仓库文件结构的代理
    all_files = set()
    for pr in repo_data.get('pullRequests', []):
        if pr and pr.get('files'):
            files_node = pr.get('files') or {}
            for file_info in files_node.get('nodes', []):
                if file_info and 'path' in file_info:
                    all_files.add(file_info['path'])

    # --- 元数据信号 ---
    # 检查仓库名称中是否包含教育或模板类关键词
    signals['name_keywords'] = {kw for kw in BOILERPLATE_KWS | EDU_KWS if kw in name}

    # 将描述和主题合并成一个长字符串，用于关键词搜索
    desc_str = description + ' '.join(topics)
    signals['desc_topics_keywords'] = {
        'app': any(kw in desc_str for kw in APP_KWS),
        'tool': any(kw in desc_str for kw in TOOL_KWS),
        'lib': any(kw in desc_str for kw in LIB_KWS),
        'edu': any(kw in desc_str for kw in EDU_KWS),
    }

    # --- 文件结构信号 ---
    signals['has_setup_or_pyproject'] = any(f in ['setup.py', 'pyproject.toml'] for f in all_files)  # 是否为Python包
    signals['has_ui_framework_file'] = any('streamlit' in f or 'gradio' in f for f in all_files)  # 是否包含UI框架文件
    signals['has_entrypoint_app'] = any(f in ['app.py', 'main.py', 'run.py'] for f in all_files)  # 是否有常见的应用入口文件
    signals['has_config_files'] = any(
        f in ['requirements.txt', 'Dockerfile', 'docker-compose.yml'] for f in all_files)  # 是否有配置文件
    signals['has_docs_folder'] = any(f.startswith('docs/') for f in all_files)  # 是否有/docs文件夹
    signals['has_examples_folder'] = any(
        f.startswith('examples/') or f.startswith('notebooks/') for f in all_files)  # 是否有/examples或/notebooks文件夹

    py_files = [f for f in all_files if f.endswith('.py')]
    signals['py_file_count'] = len(py_files)  # Python文件总数
    root_py_files = [f for f in py_files if '/' not in f]  # 位于根目录的Python文件
    # 判断是否为扁平结构：根目录Python文件占比较高
    signals['is_flat_structure'] = len(root_py_files) > 0 and len(py_files) > 0 and (
                len(root_py_files) / len(py_files) > 0.7)

    # --- 代码内容代理信号 ---
    # 统计文件路径中包含LangChain核心概念词的数量
    langchain_core_kws = {'agent', 'chain', 'prompt', 'retriever', 'llm', 'tool'}
    signals['langchain_imports_in_files'] = sum(
        1 for f in py_files if any(kw in f.lower() for kw in langchain_core_kws))

    # --- 社区与活跃度信号 ---
    stargazers = repo_data.get('stargazerCount', 0)
    forks = repo_data.get('forkCount', 0)
    created_at = pd.to_datetime(repo_data.get('createdAt'), errors='coerce')
    pushed_at = pd.to_datetime(repo_data.get('pushedAt'), errors='coerce')

    activity_score = 0
    if stargazers > 20: activity_score += 1
    if forks > 5: activity_score += 1
    if created_at and pushed_at and (pushed_at - created_at).days > 30: activity_score += 1

    if activity_score >= 3:
        signals['activity_level'] = 'High'
    elif activity_score >= 1:
        signals['activity_level'] = 'Medium'
    else:
        signals['activity_level'] = 'Low'

    return signals


# ==============================================================================
# --- 3. 决策引擎 ---
# ==============================================================================

def classify_repo(repo_name, signals):
    """
    应用多阶段分类逻辑来确定一个仓库的角色。
    """
    # --- 阶段1: 确定性分类 (处理信号非常明确的情况) ---
    if 'awesome' in signals['name_keywords']:
        return 'Knowledge/Educational'
    if signals['name_keywords'] & BOILERPLATE_KWS:  # 检查名称中是否包含模板关键词
        return 'Boilerplate/Template'
    if 'paper' in signals['name_keywords'] or 'reproducibility' in signals['name_keywords']:
        return 'Knowledge/Educational'
    if signals['py_file_count'] < 5 and signals['is_flat_structure']:  # Python文件很少且结构扁平
        return 'Uncategorized/Simple Script'

    # --- 阶段2: 基于加权评分的倾向性分类 ---
    scores = {'Application': 0, 'Infrastructure/Tool': 0, 'Library/Plugin': 0, 'Knowledge/Educational': 0}

    # 根据信号进行加权评分
    # 强信号
    if signals['has_ui_framework_file']: scores['Application'] += 10
    if signals['has_setup_or_pyproject']: scores['Library/Plugin'] += 8

    # 中等信号
    if signals['has_entrypoint_app']: scores['Application'] += 5
    if signals['desc_topics_keywords']['app']: scores['Application'] += 3
    if signals['desc_topics_keywords']['tool']: scores['Infrastructure/Tool'] += 6
    if signals['desc_topics_keywords']['lib']: scores['Library/Plugin'] += 5
    if signals['desc_topics_keywords']['edu'] or signals['has_examples_folder']: scores['Knowledge/Educational'] += 4
    if signals['has_docs_folder']:  # 有文档对库和知识传播都有利
        scores['Library/Plugin'] += 2
        scores['Knowledge/Educational'] += 2

    # 弱信号
    if signals['langchain_imports_in_files'] > 3:
        scores['Application'] += 1
        scores['Library/Plugin'] += 1

    # 负信号 (惩罚项)
    if scores['Application'] > 0 and signals['has_setup_or_pyproject']:
        scores['Application'] -= 3  # 一个被打包的项目不太可能是一个纯粹的终端应用

    if scores['Library/Plugin'] > 0 and signals['has_ui_framework_file']:
        scores['Library/Plugin'] -= 4  # 一个带UI界面的项目不太可能是一个纯粹的库

    # 返回得分最高的角色
    # 如果所有得分都为0，则归为未分类
    if all(score == 0 for score in scores.values()):
        return 'Uncategorized/Simple Script'

    primary_role = max(scores, key=scores.get)
    return primary_role


# ==============================================================================
# --- 4. & 5. 执行与输出 ---
# ==============================================================================

def main():
    """
    主函数，用于运行整个分类流程。
    """
    print("--- 仓库角色分类 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    # 排除掉仓库列表文件
    json_files = [f for f in json_files if not f.endswith('_downstream_repo_list.json')]

    results = []  # 初始化结果列表

    print(f"[*] 发现 {len(json_files)} 个仓库JSON文件待分类。")
    # 使用tqdm显示进度条
    for file_path in tqdm(json_files, desc="正在分类仓库"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        if not data or 'nameWithOwner' not in data:
            continue

        repo_name = data['nameWithOwner']
        signals = extract_signals(data)  # 提取信号

        if signals is None:
            continue

        # --- 阶段3: 最终裁定与添加辅助标签 ---
        primary_role = classify_repo(repo_name, signals)  # 获取主要角色

        # 判断辅助标签
        is_active = signals['activity_level'] in ['High', 'Medium']
        has_documentation = signals['has_docs_folder']

        # 一个简单的Monorepo（单一代码库）启发式规则：同时具有库和应用的强信号
        is_monorepo = signals['has_setup_or_pyproject'] and (
                    signals['has_ui_framework_file'] or signals['has_entrypoint_app'])

        results.append({
            'repo_name': repo_name,
            'primary_role': primary_role,
            'is_active': is_active,
            'is_monorepo': is_monorepo,
            'has_documentation': has_documentation,
            'stargazer_count': data.get('stargazerCount', 0)  # 加入star数量，方便排序和审查
        })

    # 创建DataFrame并保存结果
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'repos_classified.csv')

    # 按star数量降序排列，便于优先审查重要的仓库
    results_df.sort_values(by='stargazer_count', ascending=False, inplace=True)

    # 使用 'utf-8-sig' 编码确保在Excel中打开CSV文件时中文不会乱码
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n[✓] 分类完成！")
    print(f"[*] 结果已保存至: {output_path}")
    print("\n--- 角色分布统计 ---")
    print(results_df['primary_role'].value_counts())
    print("--------------------")


if __name__ == "__main__":
    main()