#!/usr/bin/env python3
"""
可视化模块 - 修复版
解决了 seaborn 的 label 参数冲突问题，确保能正常输出 PNG/PDF。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Visualizer:
    """RMTwin 结果可视化器"""

    def __init__(self, config):
        self.config = config
        # 使用 config 中定义的 figures 子目录
        self.output_dir = config.figure_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def _save_figure(self, fig, filename_stem):
        """核心保存函数：同时保存配置中指定的所有格式"""
        for fmt in self.config.figure_format:
            # 确保格式是 png 或 pdf
            fmt = fmt.lower().strip('.')
            path = self.output_dir / f"{filename_stem}.{fmt}"
            try:
                fig.savefig(path, format=fmt, dpi=300)
                logger.debug(f"已保存图表: {path}")
            except Exception as e:
                logger.error(f"保存图表 {filename_stem}.{fmt} 失败: {e}")
        plt.close(fig)

    def create_all_figures(self, pareto_results: pd.DataFrame,
                           baseline_results: Optional[Dict[str, pd.DataFrame]] = None,
                           optimization_history: Optional[Dict] = None):
        """生成所有主要图表"""
        logger.info(f"正在生成图表 (格式: {self.config.figure_format})...")

        try:
            self.plot_pareto_2d(pareto_results, baseline_results)
        except Exception as e:
            logger.error(f"绘制 Pareto 图失败: {e}")

        try:
            if baseline_results:
                self.create_enhanced_baseline_comparison(pareto_results, baseline_results)
        except Exception as e:
            logger.error(f"绘制基线对比图失败: {e}")

        try:
            if optimization_history:
                self.plot_convergence(optimization_history)
        except Exception as e:
            logger.error(f"绘制收敛图失败: {e}")

        logger.info(f"图表生成流程结束，检查目录: {self.output_dir}")

    def plot_pareto_2d(self, df: pd.DataFrame, baselines: Optional[Dict] = None):
        """绘制 Cost vs Recall 的 2D 散点图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制 NSGA-III (修复：移除 label 参数避免冲突)
        sns.scatterplot(
            data=df,
            x='f1_total_cost_USD',
            y='detection_recall',
            hue='f5_carbon_emissions_kgCO2e_year',
            palette='viridis',
            size='system_MTBF_hours',
            sizes=(50, 200),
            alpha=0.8,
            edgecolor='k',
            ax=ax
            # label='NSGA-III Pareto'  <-- 已移除此行
        )

        # 绘制基线
        if baselines:
            markers = ['X', 's', '^', 'D']
            colors = ['red', 'orange', 'green', 'purple']
            for i, (name, b_df) in enumerate(baselines.items()):
                if b_df is not None and not b_df.empty and 'is_feasible' in b_df.columns:
                    feasible = b_df[b_df['is_feasible']]
                    if not feasible.empty:
                        best_recall = feasible.loc[feasible['detection_recall'].idxmax()]
                        ax.scatter(
                            best_recall['f1_total_cost_USD'],
                            best_recall['detection_recall'],
                            marker=markers[i % len(markers)],
                            color=colors[i % len(colors)],
                            s=150,
                            label=f'{name} (Best)',
                            zorder=10,
                            edgecolor='white'
                        )

        ax.set_title('Pareto Front: Cost vs Detection Recall')
        ax.set_xlabel('Total Cost (USD)')
        ax.set_ylabel('Detection Recall')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 重新处理图例：将 seaborn 自动生成的图例和我们要加的基线图例合并
        # 或者简单地让 matplotlib 自动处理（seaborn 的图例通常在 ax.legend() 中自动包含）
        # 这里我们显式调用一次 legend 来包含 scatter 的 label
        if baselines:
            # 获取 seaborn 生成的 handles 和 labels
            handles, labels = ax.get_legend_handles_labels()
            # 这里的 labels 可能已经包含了 hue/size 的说明
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # 如果没有基线，保留 seaborn 默认图例位置
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # 调用通用保存函数
        self._save_figure(fig, 'pareto_cost_recall')

    def create_enhanced_baseline_comparison(self, pareto_df, baseline_results):
        """生成详细的基线对比条形图"""
        metrics = []

        # NSGA-III
        if not pareto_df.empty:
            metrics.append({'Method': 'NSGA-III', 'Metric': 'Max Recall', 'Value': pareto_df['detection_recall'].max()})
            metrics.append(
                {'Method': 'NSGA-III', 'Metric': 'Min Cost (M$)', 'Value': pareto_df['f1_total_cost_USD'].min() / 1e6})

        # Baselines
        if baseline_results:
            for name, df in baseline_results.items():
                if df is not None and not df.empty and 'is_feasible' in df.columns:
                    feasible = df[df['is_feasible']]
                    if not feasible.empty:
                        metrics.append(
                            {'Method': name, 'Metric': 'Max Recall', 'Value': feasible['detection_recall'].max()})
                        metrics.append({'Method': name, 'Metric': 'Min Cost (M$)',
                                        'Value': feasible['f1_total_cost_USD'].min() / 1e6})

        if not metrics: return

        df_metrics = pd.DataFrame(metrics)

        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data=df_metrics, x='Metric', y='Value', hue='Method')
        plt.title('Performance Comparison: NSGA-III vs Baselines')
        plt.tight_layout()

        self._save_figure(fig, 'baseline_comparison_bar')

    def plot_convergence(self, history):
        """绘制优化收敛过程"""
        if not history or 'history' not in history: return

        n_gen = len(history['history'])
        gens = range(n_gen)
        min_costs = []
        max_recalls = []

        for algo in history['history']:
            if algo.pop is not None:
                F = algo.pop.get("F")
                min_costs.append(F[:, 0].min() / 1e6)
                max_recalls.append(1 - F[:, 1].min())

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Min Cost (Million USD)', color=color)
        ax1.plot(gens, min_costs, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Max Recall', color=color)
        ax2.plot(gens, max_recalls, color=color, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Optimization Convergence')
        plt.tight_layout()

        self._save_figure(fig, 'convergence_plot')