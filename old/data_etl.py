import os
import json
import pandas as pd
import glob
from tqdm import tqdm

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================
# 输入目录: 包含原始JSON文件的文件夹
INPUT_DIR = "langchain_ecosystem_data"
# 输出目录: 用于存放处理后的Parquet文件
OUTPUT_DIR = "etl_output"


# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================

def classify_pr(files_list):
    """
    根据PR修改的文件列表，将其分类为'code', 'doc', 或 'mixed'。
    """
    if not files_list:
        return 'unknown'

    has_code = False
    has_doc = False

    code_extensions = {'.py', '.go', '.js', '.ts', '.java', '.c', '.cpp', '.sh'}
    doc_extensions = {'.md', '.mdx', '.rst', '.ipynb'}  # .ipynb can be both, but often used for docs/examples

    for file_info in files_list:
        # 确保file_info是字典并且包含'path'键
        if isinstance(file_info, dict) and 'path' in file_info:
            path = file_info['path']
            # 有些路径可能为None，需要跳过
            if path is None:
                continue

            file_ext = os.path.splitext(path)[1].lower()
            if file_ext in code_extensions or '/src/' in path or '/libs/' in path:
                has_code = True
            if file_ext in doc_extensions or '/docs/' in path or '/examples/' in path:
                has_doc = True

    if has_code and has_doc:
        return 'mixed'
    elif has_code:
        return 'code'
    elif has_doc:
        return 'doc'
    else:
        return 'other'


# ==============================================================================
# --- 主ETL逻辑 ---
# ==============================================================================

def main():
    """
    主ETL函数，负责读取、转换和加载所有数据。
    """
    print("--- LangChain 生态数据 ETL 开始 ---")

    # --- 1. 初始化 ---
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化用于存储所有数据的列表
    repos_list, users_list, issues_list, prs_list = [], [], [], []
    comments_list, star_events_list, fork_events_list = [], [], []

    # 使用集合来高效地去重用户
    seen_users = set()

    # --- 2. 提取 (Extract) ---
    # 获取所有仓库的JSON文件路径
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    # 排除掉仓库列表文件
    json_files = [f for f in json_files if not f.endswith('_downstream_repo_list.json')]

    print(f"[+] 找到 {len(json_files)} 个仓库的JSON文件，开始解析...")

    # 使用tqdm创建进度条
    for file_path in tqdm(json_files, desc="Parsing Repositories"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"\n[!] Warning: Skipping corrupted or missing file {file_path}: {e}")
            continue

        # 检查数据是否有效
        if not data:
            continue

        repo_id = data.get('id')
        repo_name = data.get('nameWithOwner')

        # --- 3. 转换 (Transform) ---

        # a. 处理仓库信息 (Repos)
        topics_data = data.get('repositoryTopics', {}) or {}
        topics = [t['topic']['name'] for t in topics_data.get('nodes', []) if t and t.get('topic')]
        repos_list.append({
            'repo_id': repo_id,
            'repo_name': repo_name,
            'stargazer_count': data.get('stargazerCount'),
            'fork_count': data.get('forkCount'),
            'disk_usage_kb': data.get('diskUsage'),
            'created_at': data.get('createdAt'),
            'pushed_at': data.get('pushedAt'),
            'description': data.get('description'),
            'url': data.get('url'),
            'topics': ','.join(topics)  # 将topics列表转换为逗号分隔的字符串
        })

        # 辅助函数：处理用户并去重
        def process_user(user_node):
            if user_node and isinstance(user_node, dict) and 'id' in user_node:
                user_id = user_node['id']
                if user_id not in seen_users:
                    seen_users.add(user_id)
                    users_list.append({
                        'user_id': user_id,
                        'login': user_node.get('login')
                    })

        # b. 处理PRs
        for pr in data.get('pullRequests', []):
            if not pr: continue
            process_user(pr.get('author'))
            process_user(pr.get('mergedBy'))

            author_node = pr.get('author') or {}
            merged_by_node = pr.get('mergedBy') or {}

            # FIX: Safely handle cases where 'files' is null
            files_node = pr.get('files') or {}
            files_list = files_node.get('nodes', [])

            prs_list.append({
                'pr_id': pr.get('id'),
                'repo_id': repo_id,
                'number': pr.get('number'),
                'title': pr.get('title'),
                'state': pr.get('state'),
                'created_at': pr.get('createdAt'),
                'closed_at': pr.get('closedAt'),
                'merged_at': pr.get('mergedAt'),
                'author_id': author_node.get('id'),
                'merged_by_id': merged_by_node.get('id'),
                'additions': pr.get('additions'),
                'deletions': pr.get('deletions'),
                'contribution_type': classify_pr(files_list)
            })

            # c. 处理PR下的评论
            for comment in pr.get('all_comments', []):
                if not comment: continue
                process_user(comment.get('author'))
                author_node = comment.get('author') or {}
                comments_list.append({
                    'comment_id': comment.get('id'),
                    'parent_id': pr.get('id'),
                    'parent_type': 'pr',
                    'repo_id': repo_id,
                    'author_id': author_node.get('id'),
                    'created_at': comment.get('createdAt')
                })

        # d. 处理Issues
        for issue in data.get('issues', []):
            if not issue: continue
            process_user(issue.get('author'))

            author_node = issue.get('author') or {}
            labels_data = issue.get('labels', {}) or {}
            labels_list = [l['name'] for l in labels_data.get('nodes', []) if l]

            issues_list.append({
                'issue_id': issue.get('id'),
                'repo_id': repo_id,
                'number': issue.get('number'),
                'title': issue.get('title'),
                'state': issue.get('state'),
                'created_at': issue.get('createdAt'),
                'closed_at': issue.get('closedAt'),
                'author_id': author_node.get('id'),
                'labels': ','.join(labels_list)
            })

            # e. 处理Issue下的评论
            for comment in issue.get('all_comments', []):
                if not comment: continue
                process_user(comment.get('author'))
                author_node = comment.get('author') or {}
                comments_list.append({
                    'comment_id': comment.get('id'),
                    'parent_id': issue.get('id'),
                    'parent_type': 'issue',
                    'repo_id': repo_id,
                    'author_id': author_node.get('id'),
                    'created_at': comment.get('createdAt')
                })

        # f. 处理星标事件
        for star in data.get('stargazers', []):
            if not star: continue
            user_node = star.get('node', {}) or {}
            process_user(user_node)
            star_events_list.append({
                'repo_id': repo_id,
                'user_id': user_node.get('id'),
                'starred_at': star.get('starredAt')
            })

        # g. 处理Fork事件
        for fork in data.get('forks', []):
            if not fork: continue
            user_node = fork.get('owner', {}) or {}
            process_user(user_node)
            fork_events_list.append({
                'fork_id': fork.get('id'),
                'repo_id': repo_id,
                'user_id': user_node.get('id'),
                'forked_at': fork.get('createdAt')
            })

    print("\n[+] 所有JSON文件解析完成。开始创建和处理DataFrames...")

    # --- 4. 加载 (Load) ---

    # a. 创建DataFrames
    repos_df = pd.DataFrame(repos_list)
    users_df = pd.DataFrame(users_list)
    issues_df = pd.DataFrame(issues_list)
    prs_df = pd.DataFrame(prs_list)
    comments_df = pd.DataFrame(comments_list)
    stars_df = pd.DataFrame(star_events_list)
    forks_df = pd.DataFrame(fork_events_list)

    # b. 统一处理时间戳
    dfs = {'repos': repos_df, 'issues': issues_df, 'prs': prs_df, 'comments': comments_df, 'stars': stars_df,
           'forks': forks_df}
    time_cols = ['created_at', 'pushed_at', 'closed_at', 'merged_at', 'starred_at', 'forked_at']

    for name, df in dfs.items():
        print(f"  -> 正在处理 {name} 表...")
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if not df.empty:
            df.drop_duplicates(inplace=True)

    # c. 保存为CSV文件
    print(f"\n[*] 开始将处理后的数据保存到 '{OUTPUT_DIR}' 文件夹...")

    # 使用 encoding='utf-8-sig' 来确保在Excel中打开时中文不会乱码
    repos_df.to_csv(os.path.join(OUTPUT_DIR, 'repos.csv'), index=False, encoding='utf-8-sig')
    users_df.to_csv(os.path.join(OUTPUT_DIR, 'users.csv'), index=False, encoding='utf-8-sig')
    issues_df.to_csv(os.path.join(OUTPUT_DIR, 'issues.csv'), index=False, encoding='utf-8-sig')
    prs_df.to_csv(os.path.join(OUTPUT_DIR, 'prs.csv'), index=False, encoding='utf-8-sig')
    comments_df.to_csv(os.path.join(OUTPUT_DIR, 'comments.csv'), index=False, encoding='utf-8-sig')
    stars_df.to_csv(os.path.join(OUTPUT_DIR, 'stars.csv'), index=False, encoding='utf-8-sig')
    forks_df.to_csv(os.path.join(OUTPUT_DIR, 'forks.csv'), index=False, encoding='utf-8-sig')

    print(f"[✓] 所有数据表已成功保存为CSV格式。")
    print("\n--- LangChain 生态数据 ETL 完成 ---")


if __name__ == "__main__":
    main()
