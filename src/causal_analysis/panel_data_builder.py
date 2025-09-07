"""
面板数据构建器
将时间序列数据转换为面板数据格式，支持固定效应分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from ..utils.logging_config import setup_logger
from config.settings import (
    FINAL_ANALYSIS_DATA_DIR, ANALYSIS_OUTPUT_DIR, 
    ANALYSIS_CONFIG, CLASSIFICATION_OUTPUT_DIR
)

# 设置日志
logger = setup_logger(__name__)


class PanelDataBuilder:
    """面板数据构建器"""
    
    def __init__(self, 
                 start_date: str = None,
                 end_date: str = None,
                 frequency: str = 'M'):
        """
        初始化面板数据构建器
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 时间频率 ('M'=月, 'Q'=季度, 'Y'=年)
        """
        self.start_date = start_date or ANALYSIS_CONFIG["start_date"]
        self.end_date = end_date or ANALYSIS_CONFIG["end_date"]
        self.frequency = frequency
        
        # 创建时间索引
        if frequency == 'M':
            self.time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        elif frequency == 'Q':
            self.time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='QS')
        elif frequency == 'Y':
            self.time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='YS')
        else:
            raise ValueError(f"不支持的频率: {frequency}")
        
        logger.info(f"初始化面板数据构建器: {len(self.time_index)} 个时间段")
    
    def build_ecosystem_panel(self) -> pd.DataFrame:
        """
        构建生态系统层面的面板数据
        创建多个维度的观测单位
        
        Returns:
            pd.DataFrame: 面板数据
        """
        logger.info("构建生态系统层面的面板数据...")
        
        # 加载月度面板数据
        monthly_data_path = ANALYSIS_OUTPUT_DIR / "monthly_panel_data.csv"
        if not monthly_data_path.exists():
            logger.error(f"月度面板数据不存在: {monthly_data_path}")
            return pd.DataFrame()
        
        monthly_data = pd.read_csv(monthly_data_path, index_col=0, parse_dates=True)
        
        # 创建多维度实体
        panel_rows = []
        
        for time_point in self.time_index:
            if time_point not in monthly_data.index:
                continue
            
            row_data = monthly_data.loc[time_point]
            
            # 1. 整体生态系统视角
            overall_data = row_data.copy()
            panel_rows.append({
                'entity_id': 'ecosystem_overall',
                'entity_type': 'comprehensive',
                'time_period': time_point,
                'focus_weight': 1.0,
                **overall_data.to_dict()
            })
            
            # 2. 代码贡献聚焦视角
            code_focused_data = row_data.copy()
            if 'mech_code_contrib' in code_focused_data:
                # 对代码相关指标给予更高权重
                code_focused_data['weighted_mech_code'] = code_focused_data['mech_code_contrib'] * 1.5
                code_focused_data['focus_intensity'] = code_focused_data['mech_code_contrib'] / max(row_data.sum(), 1)
            
            panel_rows.append({
                'entity_id': 'ecosystem_code_focus',
                'entity_type': 'code_contribution',
                'time_period': time_point,
                'focus_weight': 1.5,
                **code_focused_data.to_dict()
            })
            
            # 3. 应用开发聚焦视角
            app_focused_data = row_data.copy()
            if 'mech_app_creation' in app_focused_data:
                app_focused_data['weighted_mech_app'] = app_focused_data['mech_app_creation'] * 1.5
                app_focused_data['focus_intensity'] = app_focused_data['mech_app_creation'] / max(row_data.sum(), 1)
            
            panel_rows.append({
                'entity_id': 'ecosystem_app_focus',
                'entity_type': 'application_development',
                'time_period': time_point,
                'focus_weight': 1.5,
                **app_focused_data.to_dict()
            })
            
            # 4. 社区维护聚焦视角
            maintenance_focused_data = row_data.copy()
            if 'mech_problem_solving' in maintenance_focused_data:
                maintenance_focused_data['weighted_mech_maintenance'] = maintenance_focused_data['mech_problem_solving'] * 1.5
                maintenance_focused_data['focus_intensity'] = maintenance_focused_data['mech_problem_solving'] / max(row_data.sum(), 1)
            
            panel_rows.append({
                'entity_id': 'ecosystem_maintenance_focus',
                'entity_type': 'community_maintenance',
                'time_period': time_point,
                'focus_weight': 1.5,
                **maintenance_focused_data.to_dict()
            })
            
            # 5. 知识分享聚焦视角
            knowledge_focused_data = row_data.copy()
            if 'mech_knowledge_sharing' in knowledge_focused_data:
                knowledge_focused_data['weighted_mech_knowledge'] = knowledge_focused_data['mech_knowledge_sharing'] * 1.5
                knowledge_focused_data['focus_intensity'] = knowledge_focused_data['mech_knowledge_sharing'] / max(row_data.sum(), 1)
            
            panel_rows.append({
                'entity_id': 'ecosystem_knowledge_focus',
                'entity_type': 'knowledge_sharing',
                'time_period': time_point,
                'focus_weight': 1.5,
                **knowledge_focused_data.to_dict()
            })
        
        panel_df = pd.DataFrame(panel_rows)
        
        # 设置多重索引
        panel_df = panel_df.set_index(['entity_id', 'time_period'])
        
        logger.info(f"生态系统面板数据构建完成: {len(panel_df)} 个观测值")
        logger.info(f"实体数: {len(panel_df.index.get_level_values(0).unique())}")
        logger.info(f"时间段数: {len(panel_df.index.get_level_values(1).unique())}")
        
        return panel_df
    
    def build_repository_panel(self) -> pd.DataFrame:
        """
        构建仓库层面的面板数据
        
        Returns:
            pd.DataFrame: 仓库面板数据
        """
        logger.info("构建仓库层面的面板数据...")
        
        try:
            # 加载仓库数据
            repos_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "repos.csv")
            classified_repos_df = pd.read_csv(CLASSIFICATION_OUTPUT_DIR / "repos_classified.csv")
            
            # 合并分类信息
            repos_df = pd.merge(repos_df, classified_repos_df[['repo_name', 'primary_role']], 
                              on='repo_name', how='left')
            
            # 加载活动数据
            prs_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "prs.csv")
            issues_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "issues.csv")
            stars_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "stars.csv")
            
            # 转换时间列
            prs_df['created_at'] = pd.to_datetime(prs_df['created_at'])
            issues_df['created_at'] = pd.to_datetime(issues_df['created_at'])
            stars_df['starred_at'] = pd.to_datetime(stars_df['starred_at'])
            
            panel_rows = []
            
            # 选择活跃的仓库（有足够的活动数据）
            active_repos = self._identify_active_repositories(repos_df, prs_df, issues_df, stars_df)
            
            for repo_id in active_repos:
                repo_info = repos_df[repos_df['repo_id'] == repo_id].iloc[0]
                
                for time_point in self.time_index:
                    # 计算该时间段的仓库活动指标
                    repo_metrics = self._calculate_repo_metrics(
                        repo_id, time_point, prs_df, issues_df, stars_df
                    )
                    
                    panel_rows.append({
                        'entity_id': f"repo_{repo_id}",
                        'repo_id': repo_id,
                        'repo_name': repo_info['repo_name'],
                        'primary_role': repo_info.get('primary_role', 'unknown'),
                        'time_period': time_point,
                        **repo_metrics
                    })
            
            if not panel_rows:
                logger.warning("没有生成仓库面板数据")
                return pd.DataFrame()
            
            repo_panel_df = pd.DataFrame(panel_rows)
            repo_panel_df = repo_panel_df.set_index(['entity_id', 'time_period'])
            
            logger.info(f"仓库面板数据构建完成: {len(repo_panel_df)} 个观测值")
            
            return repo_panel_df
            
        except Exception as e:
            logger.error(f"构建仓库面板数据失败: {e}")
            return pd.DataFrame()
    
    def build_user_panel(self) -> pd.DataFrame:
        """
        构建用户层面的面板数据
        
        Returns:
            pd.DataFrame: 用户面板数据
        """
        logger.info("构建用户层面的面板数据...")
        
        try:
            # 加载用户角色数据
            user_roles_path = ANALYSIS_OUTPUT_DIR / "user_roles.csv"
            if not user_roles_path.exists():
                logger.warning("用户角色数据不存在，跳过用户面板数据构建")
                return pd.DataFrame()
            
            user_roles_df = pd.read_csv(user_roles_path)
            
            # 加载活动数据
            prs_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "prs.csv")
            issues_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "issues.csv")
            comments_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "comments.csv")
            stars_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "stars.csv")
            
            # 转换时间列
            prs_df['created_at'] = pd.to_datetime(prs_df['created_at'])
            issues_df['created_at'] = pd.to_datetime(issues_df['created_at'])
            comments_df['created_at'] = pd.to_datetime(comments_df['created_at'])
            stars_df['starred_at'] = pd.to_datetime(stars_df['starred_at'])
            
            panel_rows = []
            
            # 选择活跃用户
            active_users = self._identify_active_users(user_roles_df, prs_df, issues_df, comments_df)
            
            for user_id in active_users:
                user_info = user_roles_df[user_roles_df['user_id'] == user_id]
                if user_info.empty:
                    continue
                
                user_info = user_info.iloc[0]
                
                for time_point in self.time_index:
                    # 计算该时间段的用户活动指标
                    user_metrics = self._calculate_user_metrics(
                        user_id, time_point, prs_df, issues_df, comments_df, stars_df
                    )
                    
                    panel_rows.append({
                        'entity_id': f"user_{user_id}",
                        'user_id': user_id,
                        'login': user_info.get('login', ''),
                        'cluster': user_info.get('cluster', -1),
                        'time_period': time_point,
                        **user_metrics
                    })
            
            if not panel_rows:
                logger.warning("没有生成用户面板数据")
                return pd.DataFrame()
            
            user_panel_df = pd.DataFrame(panel_rows)
            user_panel_df = user_panel_df.set_index(['entity_id', 'time_period'])
            
            logger.info(f"用户面板数据构建完成: {len(user_panel_df)} 个观测值")
            
            return user_panel_df
            
        except Exception as e:
            logger.error(f"构建用户面板数据失败: {e}")
            return pd.DataFrame()
    
    def _identify_active_repositories(self, 
                                    repos_df: pd.DataFrame,
                                    prs_df: pd.DataFrame,
                                    issues_df: pd.DataFrame,
                                    stars_df: pd.DataFrame,
                                    min_activity_threshold: int = 5) -> List[int]:
        """识别活跃的仓库"""
        repo_activity = {}
        
        for repo_id in repos_df['repo_id']:
            activity_count = 0
            
            # PR活动
            activity_count += len(prs_df[prs_df['repo_id'] == repo_id])
            
            # Issue活动
            activity_count += len(issues_df[issues_df['repo_id'] == repo_id])
            
            # Star活动
            activity_count += len(stars_df[stars_df['repo_id'] == repo_id])
            
            if activity_count >= min_activity_threshold:
                repo_activity[repo_id] = activity_count
        
        # 返回活动最多的前50个仓库
        sorted_repos = sorted(repo_activity.keys(), key=lambda x: repo_activity[x], reverse=True)
        return sorted_repos[:50]
    
    def _identify_active_users(self,
                             user_roles_df: pd.DataFrame,
                             prs_df: pd.DataFrame,
                             issues_df: pd.DataFrame,
                             comments_df: pd.DataFrame,
                             min_activity_threshold: int = 3) -> List[int]:
        """识别活跃的用户"""
        user_activity = {}
        
        for user_id in user_roles_df['user_id']:
            activity_count = 0
            
            # PR活动
            activity_count += len(prs_df[prs_df['author_id'] == user_id])
            
            # Issue活动
            activity_count += len(issues_df[issues_df['author_id'] == user_id])
            
            # Comment活动
            activity_count += len(comments_df[comments_df['author_id'] == user_id])
            
            if activity_count >= min_activity_threshold:
                user_activity[user_id] = activity_count
        
        # 返回活动最多的前100个用户
        sorted_users = sorted(user_activity.keys(), key=lambda x: user_activity[x], reverse=True)
        return sorted_users[:100]
    
    def _calculate_repo_metrics(self,
                               repo_id: int,
                               time_point: pd.Timestamp,
                               prs_df: pd.DataFrame,
                               issues_df: pd.DataFrame,
                               stars_df: pd.DataFrame) -> Dict[str, float]:
        """计算仓库在特定时间段的活动指标"""
        # 定义时间段
        if self.frequency == 'M':
            period_start = time_point
            period_end = time_point + pd.offsets.MonthEnd(0)
        elif self.frequency == 'Q':
            period_start = time_point
            period_end = time_point + pd.offsets.QuarterEnd(0)
        else:
            period_start = time_point
            period_end = time_point + pd.offsets.YearEnd(0)
        
        # 过滤时间段内的数据
        period_prs = prs_df[
            (prs_df['repo_id'] == repo_id) &
            (prs_df['created_at'] >= period_start) &
            (prs_df['created_at'] <= period_end)
        ]
        
        period_issues = issues_df[
            (issues_df['repo_id'] == repo_id) &
            (issues_df['created_at'] >= period_start) &
            (issues_df['created_at'] <= period_end)
        ]
        
        period_stars = stars_df[
            (stars_df['repo_id'] == repo_id) &
            (stars_df['starred_at'] >= period_start) &
            (stars_df['starred_at'] <= period_end)
        ]
        
        # 计算指标
        metrics = {
            'pr_count': len(period_prs),
            'code_pr_count': len(period_prs[period_prs['contribution_type'] == 'code']),
            'doc_pr_count': len(period_prs[period_prs['contribution_type'] == 'doc']),
            'issue_count': len(period_issues),
            'star_count': len(period_stars),
            'total_additions': period_prs['additions'].sum() if not period_prs.empty else 0,
            'total_deletions': period_prs['deletions'].sum() if not period_prs.empty else 0,
            'unique_contributors': len(period_prs['author_id'].unique()) if not period_prs.empty else 0,
            'activity_intensity': len(period_prs) + len(period_issues) + len(period_stars)
        }
        
        return metrics
    
    def _calculate_user_metrics(self,
                               user_id: int,
                               time_point: pd.Timestamp,
                               prs_df: pd.DataFrame,
                               issues_df: pd.DataFrame,
                               comments_df: pd.DataFrame,
                               stars_df: pd.DataFrame) -> Dict[str, float]:
        """计算用户在特定时间段的活动指标"""
        # 定义时间段
        if self.frequency == 'M':
            period_start = time_point
            period_end = time_point + pd.offsets.MonthEnd(0)
        elif self.frequency == 'Q':
            period_start = time_point
            period_end = time_point + pd.offsets.QuarterEnd(0)
        else:
            period_start = time_point
            period_end = time_point + pd.offsets.YearEnd(0)
        
        # 过滤时间段内的数据
        period_prs = prs_df[
            (prs_df['author_id'] == user_id) &
            (prs_df['created_at'] >= period_start) &
            (prs_df['created_at'] <= period_end)
        ]
        
        period_issues = issues_df[
            (issues_df['author_id'] == user_id) &
            (issues_df['created_at'] >= period_start) &
            (issues_df['created_at'] <= period_end)
        ]
        
        period_comments = comments_df[
            (comments_df['author_id'] == user_id) &
            (comments_df['created_at'] >= period_start) &
            (comments_df['created_at'] <= period_end)
        ]
        
        period_stars = stars_df[
            (stars_df['user_id'] == user_id) &
            (stars_df['starred_at'] >= period_start) &
            (stars_df['starred_at'] <= period_end)
        ]
        
        # 计算指标
        total_contributions = len(period_prs) + len(period_issues) + len(period_comments)
        
        metrics = {
            'pr_count': len(period_prs),
            'code_pr_count': len(period_prs[period_prs['contribution_type'] == 'code']),
            'doc_pr_count': len(period_prs[period_prs['contribution_type'] == 'doc']),
            'issue_count': len(period_issues),
            'comment_count': len(period_comments),
            'star_count': len(period_stars),
            'total_contributions': total_contributions,
            'code_focus_ratio': len(period_prs[period_prs['contribution_type'] == 'code']) / max(len(period_prs), 1),
            'interaction_diversity': len([x for x in [len(period_prs), len(period_issues), len(period_comments), len(period_stars)] if x > 0]),
            'unique_repos_contributed': len(period_prs['repo_id'].unique()) if not period_prs.empty else 0
        }
        
        return metrics
    
    def save_panel_data(self, 
                       panel_data: pd.DataFrame,
                       filename: str,
                       description: str = ""):
        """
        保存面板数据到文件
        
        Args:
            panel_data: 面板数据
            filename: 文件名
            description: 数据描述
        """
        if panel_data.empty:
            logger.warning(f"面板数据为空，跳过保存: {filename}")
            return
        
        output_path = ANALYSIS_OUTPUT_DIR / filename
        panel_data.to_csv(output_path, encoding='utf-8-sig')
        
        logger.info(f"{description} 面板数据已保存至: {output_path}")
        
        # 保存数据摘要
        summary = {
            'description': description,
            'n_entities': len(panel_data.index.get_level_values(0).unique()),
            'n_time_periods': len(panel_data.index.get_level_values(1).unique()),
            'n_observations': len(panel_data),
            'time_range': {
                'start': panel_data.index.get_level_values(1).min().strftime('%Y-%m-%d'),
                'end': panel_data.index.get_level_values(1).max().strftime('%Y-%m-%d')
            },
            'variables': list(panel_data.columns),
            'entity_types': list(panel_data.index.get_level_values(0).unique()) if len(panel_data.index.get_level_values(0).unique()) < 20 else f"{len(panel_data.index.get_level_values(0).unique())} entities"
        }
        
        import json
        summary_path = ANALYSIS_OUTPUT_DIR / f"{filename.replace('.csv', '_summary.json')}"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"数据摘要已保存至: {summary_path}")


def main():
    """主函数入口"""
    logger.info("开始构建面板数据...")
    
    # 创建面板数据构建器
    builder = PanelDataBuilder(frequency='M')  # 月度频率
    
    # 1. 构建生态系统层面面板数据
    ecosystem_panel = builder.build_ecosystem_panel()
    if not ecosystem_panel.empty:
        builder.save_panel_data(ecosystem_panel, "ecosystem_panel_data.csv", "生态系统层面")
    
    # 2. 构建仓库层面面板数据
    repository_panel = builder.build_repository_panel()
    if not repository_panel.empty:
        builder.save_panel_data(repository_panel, "repository_panel_data.csv", "仓库层面")
    
    # 3. 构建用户层面面板数据
    user_panel = builder.build_user_panel()
    if not user_panel.empty:
        builder.save_panel_data(user_panel, "user_panel_data.csv", "用户层面")
    
    logger.info("面板数据构建完成！")


if __name__ == "__main__":
    main()
