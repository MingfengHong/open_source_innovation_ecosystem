import os
import json
import pandas as pd
import glob
from tqdm import tqdm
from openai import OpenAI
import time
import sys

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
# !!! 请在这里填入您的配置信息 !!!

# API 密钥 (必需)
# 警告：请将您的API密钥粘贴在此处。不要将包含密钥的文件上传到公共代码库。
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Base URL (可选, 代理或私有部署时使用)
# 如果您使用官方OpenAI服务，保持默认值 "https://api.openai.com/v1" 即可。
# 如果您使用了代理服务，请修改为您的代理地址。
BASE_URL = "https://api.openai.com/v1"

# 模型名称 (必需)
# 您可以根据需要更改为您有权访问的任何兼容模型。
# 例如: "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini"

# --- 其他配置 ---
# 输入目录: 包含原始JSON文件的文件夹
INPUT_DIR = "langchain_ecosystem_data"
# 输出目录: 用于存放LLM分类后的CSV文件
OUTPUT_DIR = "classification_output_llm"

# ==============================================================================
# --- 初始化与检查 ---
# ==============================================================================

# 在开始前，检查API密钥是否已填写
if API_KEY == "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" or not API_KEY:
    print("[-] 错误：请在脚本顶部的 'API_KEY' 变量中填写您的有效OpenAI API密钥。")
    sys.exit()  # 如果未使用有效密钥，则直接退出脚本

# 使用您定义的配置来初始化OpenAI客户端
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
except Exception as e:
    print(f"[-] 无法初始化OpenAI客户端。请检查您的API_KEY和BASE_URL配置。")
    print(f"[-] 错误: {e}")
    sys.exit()


# ==============================================================================
# --- 2. 辅助函数 ---
# ==============================================================================

def summarize_repo_for_llm(repo_data):
    # (此函数与之前版本相同，保持不变)
    """
    为LLM创建一个简洁、信息丰富的仓库摘要。
    这比将整个JSON文件扔给LLM更有效、更经济。
    """
    if not repo_data:
        return None

    name = repo_data.get('nameWithOwner', 'N/A')
    stars = repo_data.get('stargazerCount', 0)
    forks = repo_data.get('forkCount', 0)
    description = repo_data.get('description') or "No description provided."

    topics_data = repo_data.get('repositoryTopics', {}) or {}
    topics = [t['topic']['name'] for t in topics_data.get('nodes', []) if t and t.get('topic')]
    topics_str = ", ".join(topics) if topics else "No topics listed."

    all_files = set()
    for pr in repo_data.get('pullRequests', []):
        if pr and pr.get('files'):
            files_node = pr.get('files') or {}
            for file_info in files_node.get('nodes', []):
                if file_info and 'path' in file_info:
                    all_files.add(file_info['path'])
    file_sample = list(all_files)[:30]
    file_sample_str = "\n- ".join(file_sample)

    summary = f"""
Repository Name: {name}
Stars: {stars}
Forks: {forks}
Description: {description}
Topics: {topics_str}
Sample of file paths in the repository:
- {file_sample_str}
"""
    return summary


def get_llm_classification(repo_summary):
    """
    调用OpenAI API对仓库进行分类，并包含手动重试逻辑。
    使用JSON模式以确保输出格式的稳定性。
    """
    # 核心需求变更：使用英文Prompt
    system_prompt = """
You are an expert software engineering analyst specializing in open-source software ecosystems.
Your task is to accurately classify a GitHub repository based on its provided metadata.
"""

    user_prompt = f"""
Please classify the repository based on the information below.

**Classification Schema:**
You MUST choose one of the following six categories for the `primary_role`.
- **Application**: Provides a complete, runnable solution for end-users (developers or non-technical users). Its core value is in "using" it. Examples: A complete web UI chatbot, an automated data analysis service.
- **Infrastructure/Tool**: Provides services for developers to improve the development, deployment, monitoring, or evaluation lifecycle of applications. Its core value is in "enabling development". Examples: An observability platform, a testing framework for LLM apps.
- **Library/Plugin**: Provides modular, importable functionality to extend the core capabilities of a framework. Its core value is in "extending functionality". Examples: A new vector database integration, a custom Tool for a specific API.
- **Boilerplate/Template**: Primarily intended to be forked or cloned as a starting point for new projects. Its core value is in "initiating new projects".
- **Knowledge/Educational**: Primarily intended to disseminate knowledge, provide examples, or for academic research. Its core value is in "transferring information". Examples: an awesome-list, a series of tutorials, paper reproduction code.
- **Uncategorized/Simple Script**: Cannot be clearly classified into any of the above, or is just a simple script with a few code files.

**Repository Information:**
{repo_summary}

**Output Requirements:**
You MUST return only a single, valid JSON object and nothing else.
The JSON object must contain the following keys:
- `primary_role`: (string) Choose ONE of the exact strings from the **Classification Schema** above.
- `is_monorepo`: (boolean) Determine if this is a monorepo containing multiple sub-projects or distinct roles.
- `has_documentation`: (boolean) Determine if the project has dedicated documentation based on the file list (e.g., a /docs folder).
- `reasoning`: (string) Briefly explain your reasoning for the classification.

Now, analyze the information and return the JSON object.
"""

    # 核心需求变更：手动实现的重试逻辑
    max_retries = 3
    for attempt in range(max_retries + 1):  # 1次初次尝试 + 3次重试
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},  # 启用JSON模式
                temperature=0.1,
                max_tokens=500,
            )
            # 解析返回的JSON字符串
            result_json = json.loads(response.choices[0].message.content)
            # 如果成功，立即返回结果并跳出循环
            return result_json
        except Exception as e:
            # 如果发生错误，打印错误信息并准备重试
            print(f"\n[!] 调用API时发生错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                # 使用指数退避策略等待一段时间后重试
                wait_time = 2 ** attempt
                print(f"    将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 如果所有重试都失败了
                print(f"[-] 所有重试均失败，将跳过此仓库。")

    # 如果循环结束仍未成功返回，则返回一个默认的错误结构
    return {
        'primary_role': 'API_Error',
        'is_monorepo': False,
        'has_documentation': False,
        'reasoning': 'All API retries failed.'
    }


# ==============================================================================
# --- 3. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，协调整个数据采集和LLM分类流程。
    """
    print(f"--- 使用LLM ({MODEL_NAME})进行仓库角色分类 (V3) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    json_files = [f for f in json_files if not f.endswith('_downstream_repo_list.json')]

    results = []

    print(f"[*] 发现 {len(json_files)} 个仓库文件待分类。")

    for file_path in tqdm(json_files, desc="LLM正在分类仓库"):

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        if not data or 'nameWithOwner' not in data:
            continue

        repo_summary = summarize_repo_for_llm(data)
        if not repo_summary:
            continue

        llm_result = get_llm_classification(repo_summary)

        # 核心需求变更：确保输出列和格式与启发式脚本一致
        stargazers = data.get('stargazerCount', 0)
        activity_score = 0
        if stargazers > 20: activity_score += 1
        if data.get('forkCount', 0) > 5: activity_score += 1
        is_active = activity_score > 0

        results.append({
            'repo_name': data.get('nameWithOwner'),
            'primary_role': llm_result.get('primary_role', 'Parse_Error'),
            'is_active': is_active,
            'is_monorepo': llm_result.get('is_monorepo', False),
            'has_documentation': llm_result.get('has_documentation', False),
            'stargazer_count': stargazers,
            'llm_reasoning': llm_result.get('reasoning', '')
        })

    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'repos_classified_llm.csv')

    # 按star数量降序排列，便于优先审查重要的仓库
    results_df.sort_values(by='stargazer_count', ascending=False, inplace=True)

    # 核心需求变更：确保最终输出的列符合要求
    final_columns = [
        'repo_name',
        'primary_role',
        'is_active',
        'is_monorepo',
        'has_documentation',
        'stargazer_count',
        'llm_reasoning'  # 保留此列用于审计
    ]
    results_df = results_df[final_columns]

    # 使用 'utf-8-sig' 编码确保在Excel中打开CSV文件时中文不会乱码
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n[✓] LLM分类完成！")
    print(f"[*] 结果已保存至: {output_path}")
    print("\n--- LLM分类角色分布统计 ---")
    print(results_df['primary_role'].value_counts())
    print("-------------------------")


if __name__ == "__main__":
    main()