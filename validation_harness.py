#!/usr/bin/env python3
"""
Validation Harness for RMTwin (Strict Fair Comparison Mode)
Saves results to 'validation_results' folder.
"""

import logging
import numpy as np
import pandas as pd
import json
import random
import copy
from typing import List
import scipy.stats as stats

from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from utils import calculate_hypervolume, calculate_a12_effect_size, filter_nondominated

logger = logging.getLogger(__name__)


class GlobalNormalizer:
    def __init__(self):
        self.raw_feasible_points = []

    def add_run_data(self, df: pd.DataFrame):
        if df is None or len(df) == 0 or 'is_feasible' not in df.columns: return
        feasible = df[df['is_feasible'] == True]
        if len(feasible) > 0:
            cols = ['f1_total_cost_USD', 'f2_one_minus_recall', 'f3_latency_seconds',
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
                    'f6_system_reliability_inverse_MTBF']
            self.raw_feasible_points.append(feasible[cols].values)

    def compute_bounds(self):
        if not self.raw_feasible_points: return np.zeros(6), np.ones(6)
        all_points = np.vstack(self.raw_feasible_points)
        ideal = np.min(all_points, axis=0)
        nadir = np.max(all_points, axis=0)
        return ideal, np.maximum(nadir, ideal + 1e-9)


class StatisticalValidator:
    def __init__(self, ontology_graph, config, seeds: List[int] = [42, 43, 44, 45, 46]):
        self.ontology_graph = ontology_graph
        self.config = config
        self.seeds = seeds
        # 使用 ConfigManager 中定义的专用目录
        self.results_dir = config.validation_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = GlobalNormalizer()
        self.run_cache = {}

    def run_validation(self, baseline_mode: str = "fair") -> None:
        logger.info(f"START VALIDATION (Mode: {baseline_mode}, Seeds: {self.seeds})")
        logger.info(f"Results will be saved to: {self.results_dir}")

        for seed in self.seeds:
            logger.info(f"Processing Seed {seed}...")
            self._reset_random_state(seed)

            run_config = copy.deepcopy(self.config)
            run_config.random_seed = seed

            # NSGA-III
            optimizer = RMTwinOptimizer(self.ontology_graph, run_config)
            pareto_df, final_pop_df, history = optimizer.optimize()
            self._cache_run("NSGA-III", seed, final_pop_df)
            self.normalizer.add_run_data(final_pop_df)

            n_evals_actual = history.get('n_evals', run_config.population_size * run_config.n_generations)

            # Baselines
            baseline_runner = BaselineRunner(self.ontology_graph, run_config)
            bl_results = baseline_runner.run_all_methods(mode=baseline_mode, seed=seed, eval_budget=n_evals_actual)
            for name, df in bl_results.items():
                self._cache_run(name, seed, df)
                self.normalizer.add_run_data(df)

        logger.info("Computing global bounds...")
        ideal, nadir = self.normalizer.compute_bounds()

        all_metrics = []
        for method in self.run_cache:
            for seed in self.run_cache[method]:
                df = self.run_cache[method][seed]
                metrics = self._compute_metrics(df, method, seed, ideal, nadir)
                all_metrics.append(metrics)
                # 保存每次运行的 CSV 数据
                df.to_csv(self.results_dir / f"run_{method.replace(' ', '_')}_seed{seed}.csv", index=False)

        self._generate_reports(all_metrics)

    def _reset_random_state(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _cache_run(self, method, seed, df):
        if method not in self.run_cache: self.run_cache[method] = {}
        self.run_cache[method][seed] = df

    def _compute_metrics(self, df, method, seed, ideal, nadir):
        if df is None or len(df) == 0: return self._empty_metrics(method, seed)
        total = len(df)
        feasible_df = df[df['is_feasible'] == True]
        n_feasible = len(feasible_df)
        feasibility_rate = n_feasible / total if total > 0 else 0

        if n_feasible == 0:
            m = self._empty_metrics(method, seed)
            m['Runtime'] = df['time_seconds'].iloc[0] if 'time_seconds' in df else 0
            return m

        obj_cols = ['f1_total_cost_USD', 'f2_one_minus_recall', 'f3_latency_seconds',
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
                    'f6_system_reliability_inverse_MTBF']
        F = feasible_df[obj_cols].values
        F_nd = filter_nondominated(F)
        F_norm = (F_nd - ideal) / (nadir - ideal)

        hv = calculate_hypervolume(F_norm, ref_point=np.ones(6) * 1.1)
        max_recall = feasible_df['detection_recall'].max()
        norm_cost = (feasible_df['f1_total_cost_USD'] / 1e7).mean()
        runtime = df['time_seconds'].mean() if 'time_seconds' in df else 0

        return {"Method": method, "Seed": seed, "Feasibility_Rate": feasibility_rate,
                "HV": hv, "Max_Recall": max_recall, "Norm_Cost": norm_cost, "Runtime": runtime}

    def _empty_metrics(self, method, seed):
        return {"Method": method, "Seed": seed, "Feasibility_Rate": 0.0, "HV": 0.0,
                "Max_Recall": 0.0, "Norm_Cost": np.nan, "Runtime": 0.0}

    def _generate_reports(self, metrics_list):
        df = pd.DataFrame(metrics_list)
        df.to_csv(self.results_dir / "all_metrics_raw.csv", index=False)
        summary = df.groupby("Method").agg(["mean", "std"])
        summary.to_csv(self.results_dir / "metrics_summary.csv")
        logger.info("\nValidation Summary:\n" + str(summary))

        stats_out = {}
        target, baseline = "NSGA-III", "Random Search"
        if target in df['Method'].values and baseline in df['Method'].values:
            hv_target = df[df['Method'] == target]['HV'].values
            hv_base = df[df['Method'] == baseline]['HV'].values
            try:
                u_stat, p_val = stats.mannwhitneyu(hv_target, hv_base, alternative='two-sided')
                a12 = calculate_a12_effect_size(hv_target, hv_base)
                stats_out[baseline] = {"p_value": p_val, "A12": a12}
            except Exception as e:
                logger.warning(f"Stats error: {e}")

        with open(self.results_dir / "stats_tests.json", "w") as f:
            json.dump(stats_out, f, indent=2)