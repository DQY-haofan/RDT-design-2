#!/usr/bin/env python3
"""
核心优化模块 - NSGA-III实现 (Step 1.1 加固版)
统一接口三元组返回，记录真实种群大小和执行时间
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Tuple, Dict, List, Optional

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

logger = logging.getLogger(__name__)


class RMTwinProblem(Problem):
    def __init__(self, evaluator):
        super().__init__(n_var=11, n_obj=6, n_constr=5, xl=0, xu=1)
        self.evaluator = evaluator
        self._eval_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        objectives, constraints = self.evaluator.evaluate_batch(X)
        out["F"] = objectives
        out["G"] = constraints
        self._eval_count += len(X)
        if self._eval_count % 1000 == 0:
            logger.debug(f"已评估 {self._eval_count} 个解决方案...")


class RMTwinOptimizer:
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        from fitness_evaluation import EnhancedFitnessEvaluatorV3
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)
        self.problem = RMTwinProblem(self.evaluator)
        self.algorithm, self.pop_size_used = self._configure_algorithm()

    def _configure_algorithm(self):
        if self.config.n_objectives <= 2:
            algo = NSGA2(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, prob=1.0 / self.problem.n_var),
                eliminate_duplicates=True
            )
            return algo, self.config.population_size
        else:
            try:
                ref_dirs = get_reference_directions(
                    "das-dennis",
                    self.config.n_objectives,
                    n_partitions=self.config.n_partitions
                )
            except Exception as e:
                logger.error(f"参考方向失败: {e}，回退默认设置")
                ref_dirs = get_reference_directions("das-dennis", self.config.n_objectives, n_partitions=4)

            min_pop_size = len(ref_dirs)
            pop_size = max(self.config.population_size, min_pop_size)
            if pop_size % 4 != 0: pop_size = ((pop_size // 4) + 1) * 4

            logger.info(f"配置 NSGA-III: Pop={pop_size}, RefDirs={len(ref_dirs)}")

            algo = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, prob=1.0 / self.problem.n_var),
                eliminate_duplicates=True
            )
            return algo, pop_size

    def optimize(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        logger.info(f"开始优化 (Seed={self.config.random_seed})...")

        termination = get_termination("n_gen", self.config.n_generations)

        start_time = time.time()
        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=self.config.random_seed,
            save_history=True,
            verbose=True
        )
        total_exec_time = time.time() - start_time

        # 1. Pareto Front
        pareto_df = self._process_solutions(res.X, res.F, res.G, total_exec_time)

        # 2. Final Population
        final_pop_df = pd.DataFrame()
        if res.pop is not None:
            final_pop_df = self._process_solutions(
                res.pop.get("X"), res.pop.get("F"), res.pop.get("G"), total_exec_time
            )

        history = {
            'n_evals': self.problem._eval_count,
            'exec_time': res.exec_time if hasattr(res, 'exec_time') else total_exec_time,
            'n_gen': len(res.history) if res.history else 0,
            'pop_size_used': self.pop_size_used,  # 用于 Baseline Fair Mode
            'history': res.history
        }

        return pareto_df, final_pop_df, history

    def _process_solutions(self, X, F, G, exec_time) -> pd.DataFrame:
        if X is None or len(X) == 0: return pd.DataFrame()

        X = np.atleast_2d(X)
        F = np.atleast_2d(F)
        if G is None: _, G = self.evaluator.evaluate_batch(X)
        G = np.atleast_2d(G)

        results = []
        for i in range(len(X)):
            config = self.evaluator.solution_mapper.decode_solution(X[i])
            is_feasible = np.all(G[i] <= 1e-6)

            entry = {
                'solution_id': i,
                'f1_total_cost_USD': float(F[i, 0]),
                'f2_one_minus_recall': float(F[i, 1]),
                'f3_latency_seconds': float(F[i, 2]),
                'f4_traffic_disruption_hours': float(F[i, 3]),
                'f5_carbon_emissions_kgCO2e_year': float(F[i, 4]),
                'f6_system_reliability_inverse_MTBF': float(F[i, 5]),
                'g1': float(G[i, 0]), 'g2': float(G[i, 1]), 'g3': float(G[i, 2]),
                'g4': float(G[i, 3]), 'g5': float(G[i, 4]),
                'detection_recall': float(1 - F[i, 1]),
                'system_MTBF_hours': float(1 / F[i, 5] if F[i, 5] > 0 else 1e6),
                'annual_cost_USD': float(F[i, 0] / self.config.planning_horizon_years),
                'is_feasible': bool(is_feasible),
                'time_seconds': exec_time,  # 统一时间字段
                'sensor': config['sensor'].split('#')[-1],
                'algorithm': config['algorithm'].split('#')[-1]
            }
            results.append(entry)
        return pd.DataFrame(results)