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
# 输出目录: 用于存放相关性判断结果的CSV文件
OUTPUT_DIR = "relevance_judgment_output"

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


def get_relevance_judgment(repo_summary):
    """
    调用OpenAI API判断仓库是否与研究主题相关，并包含手动重试逻辑。
    """
    # 核心需求变更：全新的英文Prompt，用于进行布尔值判断
    system_prompt = """
You are a highly intelligent research assistant for a study on the LangChain open-source ecosystem.
Your task is to determine if a GitHub repository is relevant for this study based on its goals. You must make a strict TRUE or FALSE judgment.
"""

    user_prompt = f"""
Please determine if the repository described below is relevant for my research study.

**Research Context:**
My study investigates the core dynamics of the LangChain ecosystem. I am NOT interested in every project that simply uses LangChain. I am ONLY interested in projects that can help answer these theoretical questions:
1.  **Ecosystem Evolution**: How does the ecosystem's structure evolve?
2.  **Role Symbiosis**: How do different developer roles (e.g., application builders vs. tool creators) depend on each other?
3.  **Value Creation**: How do different activities (e.g., building apps vs. building tools) contribute to the ecosystem's overall health and attractiveness?

**Criteria for Relevance (Mark as TRUE):**
A repository is relevant if it falls into one of these categories:
1.  **Direct Downstream Project**: It uses LangChain as a core component to build a meaningful application, library, or plugin. It represents a genuine "act of creation" within the ecosystem.
2.  **Ecosystem-Problem Solver**: It is a tool (for things like monitoring, evaluation, deployment, observability) created to solve common problems that arise when developing with LangChain. These tools are crucial for the ecosystem's robustness.
3.  **Significant Integration**: It is a pre-existing platform or tool that has made supporting or integrating LangChain a significant, highlighted part of its functionality, indicating a symbiotic relationship.

**Criteria for Irrelevance (Mark as FALSE):**
A repository is irrelevant if it is:
1.  A trivial "toy project", a simple script, a student homework assignment, or a list of bookmarks (e.g., awesome-lists).
2.  A project that only mentions LangChain in passing in its documentation but has no meaningful code integration.
3.  A large platform where LangChain is just one of hundreds of minor, non-core plugins and not a strategic focus.

**Repository Information:**
{repo_summary}

**Output Requirements:**
You MUST return only a single, valid JSON object and nothing else.
The JSON object must contain the following keys:
- `is_relevant_to_study`: (boolean) Your final judgment, either `true` or `false`.
- `reasoning`: (string) A brief, one-sentence explanation for your decision, referencing the criteria above.

Now, based on the research context and criteria, provide your JSON judgment.
"""

    # 手动实现的重试逻辑 (与之前版本相同)
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,  # 对于分类任务，使用0温性以获得最稳定的结果
                max_tokens=300,
            )
            result_json = json.loads(response.choices[0].message.content)
            return result_json
        except Exception as e:
            print(f"\n[!] 调用API时发生错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"    将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[-] 所有重试均失败，将跳过此仓库。")

    return {
        'is_relevant_to_study': False,  # 出错时，默认为FALSE，避免污染数据集
        'reasoning': 'All API retries failed.'
    }


# ==============================================================================
# --- 3. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，协调整个数据采集和LLM判断流程。
    """
    print(f"--- 使用LLM ({MODEL_NAME}) 判断仓库的研究相关性 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    json_files = [f for f in json_files if not f.endswith('_downstream_repo_list.json')]

    results = []

    print(f"[*] 发现 {len(json_files)} 个仓库文件待判断。")

    for file_path in tqdm(json_files, desc="LLM正在判断相关性"):

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

        llm_result = get_relevance_judgment(repo_summary)

        results.append({
            'repo_name': data.get('nameWithOwner'),
            'stargazer_count': data.get('stargazerCount', 0),
            'is_relevant_to_study': llm_result.get('is_relevant_to_study', False),  # 核心输出
            'llm_reasoning': llm_result.get('reasoning', '')
        })


    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'repo_relevance_judgment.csv')

    results_df.sort_values(by='stargazer_count', ascending=False, inplace=True)

    # 使用 'utf-8-sig' 编码确保在Excel中打开CSV文件时中文不会乱码
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n[✓] 相关性判断完成！")
    print(f"[*] 结果已保存至: {output_path}")
    print("\n--- 相关性分布统计 ---")
    print(results_df['is_relevant_to_study'].value_counts())
    print("----------------------")


if __name__ == "__main__":
    main()