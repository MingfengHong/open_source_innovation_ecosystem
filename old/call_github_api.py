import os
import requests
import json
import time
from datetime import datetime, timedelta
import calendar

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================
# !!! 警告：请将您的GitHub Personal Access Token粘贴在此处 !!!
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN_HERE"
API_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}
DATA_DIR = "langchain_ecosystem_data"

# ==============================================================================
# --- GraphQL 查询语句 (已重构和扩展) ---
# ==============================================================================

# 查询1: 搜索符合条件的下游仓库 (保持不变)
SEARCH_REPOS_QUERY = """
query SearchDownstreamRepos($searchQuery: String!, $cursor: String) {
  search(query: $searchQuery, type: REPOSITORY, first: 100, after: $cursor) {
    repositoryCount
    pageInfo {
      endCursor
      hasNextPage
    }
    nodes {
      ... on Repository {
        nameWithOwner
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# 查询2: 获取仓库的基础信息和各项总数 (已扩展)
REPO_BASE_INFO_QUERY = """
query RepoBaseInfo($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    nameWithOwner
    stargazerCount
    forkCount
    diskUsage
    createdAt
    pushedAt
    description
    url
    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }
    pullRequests {
      totalCount
    }
    issues {
      totalCount
    }
    stargazers {
        totalCount
    }
    forks {
        totalCount
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# 查询3: 分页获取Pull Requests (已扩展)
PAGINATED_PRS_QUERY = """
query PaginatedPrs($owner: String!, $name: String!, $prCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $prCursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { endCursor, hasNextPage }
      nodes {
        id
        number
        title
        state
        createdAt
        closedAt
        mergedAt
        additions
        deletions
        author { ... on User { id, login } }
        mergedBy { ... on User { id, login } }
        files(first: 100) {
          nodes { path }
        }
        closingIssuesReferences(first: 10) {
          nodes { number }
        }
        comments(first: 1) {
            totalCount
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# 查询4: 分页获取Issues (已扩展)
PAGINATED_ISSUES_QUERY = """
query PaginatedIssues($owner: String!, $name: String!, $issueCursor: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 50, after: $issueCursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { endCursor, hasNextPage }
      nodes {
        id
        number
        title
        state
        createdAt
        closedAt
        author { ... on User { id, login } }
        labels(first: 10) {
          nodes { name, color }
        }
        comments(first: 1) {
          totalCount
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# 新增查询5: 分页获取评论 (用于Issue和PR) - 已修复
PAGINATED_COMMENTS_QUERY = """
query PaginatedComments($nodeId: ID!, $commentCursor: String) {
    node(id: $nodeId) {
        ... on Issue {
            comments(first: 100, after: $commentCursor) {
                pageInfo { endCursor, hasNextPage }
                nodes {
                    id
                    author { ... on User { id, login } }
                    createdAt
                    bodyText
                }
            }
        }
        ... on PullRequest {
            comments(first: 100, after: $commentCursor) {
                pageInfo { endCursor, hasNextPage }
                nodes {
                    id
                    author { ... on User { id, login } }
                    createdAt
                    bodyText
                }
            }
        }
    }
    rateLimit {
        remaining
        resetAt
    }
}
"""

# 新增查询6: 分页获取带时间戳的星标事件
PAGINATED_STARGAZERS_QUERY = """
query PaginatedStargazers($owner: String!, $name: String!, $starCursor: String) {
    repository(owner: $owner, name: $name) {
        stargazers(first: 100, after: $starCursor, orderBy: {field: STARRED_AT, direction: ASC}) {
            pageInfo { endCursor, hasNextPage }
            edges {
                starredAt
                node {
                    id
                    login
                }
            }
        }
    }
    rateLimit {
        remaining
        resetAt
    }
}
"""

# 新增查询7: 分页获取Fork事件
PAGINATED_FORKS_QUERY = """
query PaginatedForks($owner: String!, $name: String!, $forkCursor: String) {
    repository(owner: $owner, name: $name) {
        forks(first: 100, after: $forkCursor, orderBy: {field: CREATED_AT, direction: ASC}) {
            pageInfo { endCursor, hasNextPage }
            nodes {
                id
                nameWithOwner
                createdAt
                owner {
                    id
                    login
                }
            }
        }
    }
    rateLimit {
        remaining
        resetAt
    }
}
"""


# ==============================================================================
# --- 核心功能函数 ---
# ==============================================================================

def run_query(query, variables):
    """
    执行GraphQL查询并处理速率限制。
    如果遇到速率限制，会自动等待到重置时间。
    """
    while True:
        payload = {"query": query, "variables": variables}
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                print(f"[-] GraphQL 查询错误: {result['errors']}")
                return None

            rate_limit = result.get("data", {}).get("rateLimit", {})
            if rate_limit and rate_limit["remaining"] < 100:
                reset_at = datetime.fromisoformat(rate_limit["resetAt"].replace("Z", "+00:00"))
                now = datetime.now(reset_at.tzinfo)
                wait_time = (reset_at - now).total_seconds() + 10
                if wait_time > 0:
                    print(f"[!] 速率限制接近耗尽，剩余 {rate_limit['remaining']}。等待 {int(wait_time)} 秒...")
                    time.sleep(wait_time)

            return result

        except requests.exceptions.RequestException as e:
            print(f"[-] 网络请求异常: {e}。10秒后重试...")
            time.sleep(10)


def fetch_all_downstream_repos():
    """
    获取所有下游仓库列表，并保存到文件。
    通过按月分段搜索来绕过GitHub API的1000个结果上限。
    """
    repo_list_file = os.path.join(DATA_DIR, "_downstream_repo_list.json")
    if os.path.exists(repo_list_file):
        print(f"[+] 从缓存文件 '{repo_list_file}' 中加载下游仓库列表。")
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("[*] 开始分时段搜索下游仓库以绕过1000结果上限...")
    all_repos_set = set()

    start_year = 2022
    current_year = datetime.now().year

    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            if year == 2022 and month < 11:
                continue
            if year == current_year and month > datetime.now().month:
                break

            _, num_days = calendar.monthrange(year, month)
            start_date_str = f"{year}-{month:02d}-01"
            end_date_str = f"{year}-{month:02d}-{num_days:02d}"
            date_range = f"created:{start_date_str}..{end_date_str}"

            print(f"\n--- 正在搜索时间段: {date_range} ---")

            base_query = "langchain in:readme,description,topics stars:>10 forks:>2 fork:false"
            search_query_with_date = f"{base_query} {date_range} sort:updated-desc"

            has_next_page = True
            cursor = None
            page_count = 1

            while has_next_page:
                print(f"[*] 正在获取第 {page_count} 页搜索结果...")
                variables = {"searchQuery": search_query_with_date, "cursor": cursor}
                result = run_query(SEARCH_REPOS_QUERY, variables)
                if not result or "data" not in result or not result["data"]["search"]:
                    print("[-] 获取搜索结果失败或结果为空，跳至下一时间段。")
                    break

                search_data = result["data"]["search"]

                if search_data["repositoryCount"] > 950:
                    print(
                        f"[!] 警告: 时间段 {date_range} 的结果数 ({search_data['repositoryCount']}) 接近1000，可能存在数据丢失。")

                repos_on_page = [node["nameWithOwner"] for node in search_data["nodes"]]
                all_repos_set.update(repos_on_page)

                page_info = search_data["pageInfo"]
                has_next_page = page_info["hasNextPage"]
                cursor = page_info["endCursor"]

                rate_limit = result.get("data", {}).get("rateLimit", {})
                print(f"  > 已获取 {len(repos_on_page)} 个仓库，当前时段总计 {search_data['repositoryCount']}。")
                print(f"  > API速率限制剩余: {rate_limit.get('remaining', 'N/A')}")

                page_count += 1
                time.sleep(1)

    all_repos = sorted(list(all_repos_set))
    print(f"\n[+] 搜索完成！共找到 {len(all_repos)} 个不重复的下游仓库。")
    with open(repo_list_file, 'w', encoding='utf-8') as f:
        json.dump(all_repos, f, ensure_ascii=False, indent=2)
    print(f"[+] 仓库列表已保存至 '{repo_list_file}'。")
    return all_repos


def fetch_all_comments(node_id):
    """
    为给定的Issue或PR ID获取所有评论。
    """
    all_comments = []
    has_next_page = True
    cursor = None
    while has_next_page:
        variables = {"nodeId": node_id, "commentCursor": cursor}
        result = run_query(PAGINATED_COMMENTS_QUERY, variables)
        if not result or not result.get("data", {}).get("node"):
            break

        # The response structure for comments is consistent thanks to GraphQL fragments
        node_data = result["data"]["node"]
        if "comments" not in node_data or not node_data["comments"]:
            break  # No comments found or empty comments object

        comment_data = node_data["comments"]
        all_comments.extend(comment_data["nodes"])
        page_info = comment_data["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"]
    return all_comments


def fetch_repo_details(owner, name):
    """
    获取单个仓库的全部详细信息，包括所有分页数据。
    """
    print(f"\n[*] 开始处理仓库: {owner}/{name}")

    # 1. 获取基础信息和总数
    print("  [->] 获取仓库基础信息...")
    base_info_vars = {"owner": owner, "name": name}
    base_result = run_query(REPO_BASE_INFO_QUERY, base_info_vars)
    if not base_result or not base_result.get("data", {}).get("repository"):
        print(f"  [-] 无法获取仓库 {owner}/{name} 的基础信息，跳过。")
        return None

    repo_data = base_result["data"]["repository"]
    total_pr_count = repo_data.get("pullRequests", {}).get("totalCount", 0)
    total_issue_count = repo_data.get("issues", {}).get("totalCount", 0)
    total_stargazer_count = repo_data.get("stargazers", {}).get("totalCount", 0)
    total_fork_count = repo_data.get("forks", {}).get("totalCount", 0)

    # 初始化数据列表
    repo_data["pullRequests"], repo_data["issues"], repo_data["stargazers"], repo_data["forks"] = [], [], [], []
    print(
        f"  [✓] 基础信息获取完成。PRs: {total_pr_count}, Issues: {total_issue_count}, Stars: {total_stargazer_count}, Forks: {total_fork_count}")

    # 2. 分页获取所有Pull Requests
    if total_pr_count > 0:
        print("  [->] 获取 Pull Requests...")
        has_next_page, cursor = True, None
        while has_next_page:
            pr_vars = {"owner": owner, "name": name, "prCursor": cursor}
            pr_result = run_query(PAGINATED_PRS_QUERY, pr_vars)
            if not pr_result or not pr_result.get("data", {}).get("repository"): break
            pr_data = pr_result["data"]["repository"]["pullRequests"]

            # 为每个PR获取其所有评论
            for pr_node in pr_data["nodes"]:
                comment_count = pr_node.get("comments", {}).get("totalCount", 0)
                if comment_count > 0:
                    print(f"    [->] 获取 PR #{pr_node['number']} 的 {comment_count} 条评论...")
                    pr_node["all_comments"] = fetch_all_comments(pr_node["id"])

            repo_data["pullRequests"].extend(pr_data["nodes"])
            page_info = pr_data["pageInfo"]
            has_next_page, cursor = page_info["hasNextPage"], page_info["endCursor"]
            print(f"    > 已获取 {len(repo_data['pullRequests'])} / {total_pr_count} 个PRs...")

    # 3. 分页获取所有Issues
    if total_issue_count > 0:
        print("  [->] 获取 Issues...")
        has_next_page, cursor = True, None
        while has_next_page:
            issue_vars = {"owner": owner, "name": name, "issueCursor": cursor}
            issue_result = run_query(PAGINATED_ISSUES_QUERY, issue_vars)
            if not issue_result or not issue_result.get("data", {}).get("repository"): break
            issue_data = issue_result["data"]["repository"]["issues"]

            # 为每个Issue获取其所有评论
            for issue_node in issue_data["nodes"]:
                comment_count = issue_node.get("comments", {}).get("totalCount", 0)
                if comment_count > 0:
                    print(f"    [->] 获取 Issue #{issue_node['number']} 的 {comment_count} 条评论...")
                    issue_node["all_comments"] = fetch_all_comments(issue_node["id"])

            repo_data["issues"].extend(issue_data["nodes"])
            page_info = issue_data["pageInfo"]
            has_next_page, cursor = page_info["hasNextPage"], page_info["endCursor"]
            print(f"    > 已获取 {len(repo_data['issues'])} / {total_issue_count} 个Issues...")

    # 4. 分页获取所有Stargazer事件
    if total_stargazer_count > 0:
        print("  [->] 获取 Stargazer 事件...")
        has_next_page, cursor = True, None
        while has_next_page:
            star_vars = {"owner": owner, "name": name, "starCursor": cursor}
            star_result = run_query(PAGINATED_STARGAZERS_QUERY, star_vars)
            if not star_result or not star_result.get("data", {}).get("repository"): break
            star_data = star_result["data"]["repository"]["stargazers"]
            repo_data["stargazers"].extend(star_data["edges"])
            page_info = star_data["pageInfo"]
            has_next_page, cursor = page_info["hasNextPage"], page_info["endCursor"]
            print(f"    > 已获取 {len(repo_data['stargazers'])} / {total_stargazer_count} 个Stargazer事件...")

    # 5. 分页获取所有Fork事件
    if total_fork_count > 0:
        print("  [->] 获取 Fork 事件...")
        has_next_page, cursor = True, None
        while has_next_page:
            fork_vars = {"owner": owner, "name": name, "forkCursor": cursor}
            fork_result = run_query(PAGINATED_FORKS_QUERY, fork_vars)
            if not fork_result or not fork_result.get("data", {}).get("repository"): break
            fork_data = fork_result["data"]["repository"]["forks"]
            repo_data["forks"].extend(fork_data["nodes"])
            page_info = fork_data["pageInfo"]
            has_next_page, cursor = page_info["hasNextPage"], page_info["endCursor"]
            print(f"    > 已获取 {len(repo_data['forks'])} / {total_fork_count} 个Fork事件...")

    return repo_data


# ==============================================================================
# --- 主执行逻辑 (保持不变) ---
# ==============================================================================
def main():
    """
    主函数，协调整个数据采集流程。
    """
    print("--- LangChain生态系统数据采集器 (V3 - 增强版) ---")

    if GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print("\n[错误] 请在脚本顶部设置您的 GITHUB_TOKEN！")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    target_repos = fetch_all_downstream_repos()

    official_repo = "langchain-ai/langchain"
    if official_repo not in target_repos:
        target_repos.insert(0, official_repo)

    print(f"\n[*] 准备开始处理总计 {len(target_repos)} 个仓库。")

    for i, repo_full_name in enumerate(target_repos):
        print(f"\n{'=' * 60}\n处理进度: [{i + 1}/{len(target_repos)}] - {repo_full_name}\n{'=' * 60}")

        output_filename = os.path.join(DATA_DIR, f"{repo_full_name.replace('/', '_')}.json")
        if os.path.exists(output_filename):
            print(f"[✓] 仓库 {repo_full_name} 的数据已存在，跳过。")
            continue

        try:
            owner, name = repo_full_name.split('/')
        except ValueError:
            print(f"[-] 仓库名称格式不正确: {repo_full_name}，跳过。")
            continue

        detailed_data = fetch_repo_details(owner, name)

        if detailed_data:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, ensure_ascii=False, indent=2)
            print(f"[+] 仓库 {repo_full_name} 的数据已成功保存至 '{output_filename}'")
        else:
            print(f"[-] 未能获取仓库 {repo_full_name} 的详细数据，已跳过。")

    print("\n\n--- 所有采集任务已完成 ---")


if __name__ == "__main__":
    main()
