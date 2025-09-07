"""
API配置文件
管理各种API的配置信息，包括GitHub API和OpenAI API
"""

import os
from typing import Optional

class APIConfig:
    """API配置管理类"""
    
    def __init__(self):
        self.github_token = self._get_env_var("GITHUB_TOKEN", "xxx")
        self.openai_api_key = self._get_env_var("OPENAI_API_KEY", "xxx")
        self.openai_base_url = self._get_env_var("OPENAI_BASE_URL", "https://xxx/v1")
        self.openai_model = self._get_env_var("OPENAI_MODEL", "xxx")
    
    def _get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """从环境变量获取配置，如果不存在则使用默认值"""
        return os.getenv(key, default)
    
    @property
    def github_headers(self) -> dict:
        """GitHub API请求头"""
        return {
            "Authorization": f"bearer {self.github_token}",
            "Content-Type": "application/json"
        }
    
    @property
    def github_api_url(self) -> str:
        """GitHub GraphQL API URL"""
        return "https://api.github.com/graphql"
    
    def validate_github_token(self) -> bool:
        """验证GitHub Token是否有效"""
        return self.github_token is not None and len(self.github_token) > 20
    
    def validate_openai_key(self) -> bool:
        """验证OpenAI API Key是否有效"""
        return self.openai_api_key is not None and self.openai_api_key.startswith("sk-")

# 全局API配置实例
api_config = APIConfig()
