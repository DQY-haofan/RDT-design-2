#!/usr/bin/env python3
"""
Baseline Methods for RMTwin (Step 1.1)
Clean naming, consistent time tracking, and fair budget support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import time
from abc import ABC, abstractmethod

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.results = []
        self.execution_time = 0

    @abstractmethod
    def optimize(self, seed: int, **kwargs) -> pd.DataFrame:
        pass

    def _create_result_entry(self, x: np.ndarray, obj: np.ndarray, constr: np.ndarray, sol_id: int) -> Dict:
        is_feasible = np.all(constr <= 1e-6)
        config = self.evaluator.solution_mapper.decode_solution(x)
        return {
            'solution_id': sol_id,
            'method': self.__class__.__name__,
            'f1_total_cost_USD': float(obj[0]),
            'f2_one_minus_recall': float(obj[1]),
            'f3_latency_seconds': float(obj[2]),
            'f4_traffic_disruption_hours': float(obj[3]),
            'f5_carbon_emissions_kgCO2e_year': float(obj[4]),
            'f6_system_reliability_inverse_MTBF': float(obj[5]),
            'g1': float(constr[0]), 'g2': float(constr[1]), 'g3': float(constr[2]),
            'g4': float(constr[3]), 'g5': float(constr[4]),
            'detection_recall': float(1 - obj[1]),
            'system_MTBF_hours': float(1 / obj[5] if obj[5] > 0 else 1e6),
            'is_feasible': bool(is_feasible),
            'time_seconds': self.execution_time,  # 统一时间字段
            'sensor': config['sensor'].split('#')[-1],
            'algorithm': config['algorithm'].split('#')[-1]
        }


class RandomSearchUniformBaseline(BaselineMethod):
    def optimize(self, seed: int, n_samples: int = 3000) -> pd.DataFrame:
        logger.info(f"Running Random Search (Seed={seed}, N={n_samples})")
        rng = np.random.default_rng(seed)
        start_time = time.time()

        X = rng.random((n_samples, 11))
        objs, constrs = self.evaluator.evaluate_batch(X)

        self.execution_time = time.time() - start_time  # Record execution time BEFORE loop

        for i in range(n_samples):
            self.results.append(self._create_result_entry(X[i], objs[i], constrs[i], i))

        # Update time in all records
        for r in self.results: r['time_seconds'] = self.execution_time
        return pd.DataFrame(self.results)


class WeightedSumNSGA2Baseline(BaselineMethod):
    """NSGA-II on scalarized problems."""

    class ScalarizedProblem(Problem):
        def __init__(self, evaluator, weights):
            super().__init__(n_var=11, n_obj=1, n_constr=5, xl=0, xu=1)
            self.evaluator = evaluator
            self.weights = weights
            self.norm_bounds = np.array([
                [1e5, 5e6], [0, 0.5], [0.1, 300], [0, 300], [500, 5e4], [0, 1e-3]
            ])

        def _evaluate(self, X, out, *args, **kwargs):
            objs, constrs = self.evaluator.evaluate_batch(X)
            norm_objs = (objs - self.norm_bounds[:, 0]) / (self.norm_bounds[:, 1] - self.norm_bounds[:, 0])
            norm_objs = np.clip(norm_objs, 0, 1)
            f_scalar = np.sum(norm_objs * self.weights, axis=1)
            out["F"] = f_scalar[:, None]
            out["G"] = constrs

    def optimize(self, seed: int, population_size=100, n_generations=100) -> pd.DataFrame:
        logger.info(f"Running Weighted Sum (Seed={seed})")
        start_time = time.time()

        weight_vectors = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([1 / 6] * 6),
            np.array([0.5, 0.5, 0, 0, 0, 0])
        ]

        cnt = 0
        for idx, w in enumerate(weight_vectors):
            problem = self.ScalarizedProblem(self.evaluator, w)
            algorithm = NSGA2(
                pop_size=population_size, sampling=FloatRandomSampling(),
                crossover=SBX(eta=20, prob=0.9), mutation=PM(eta=20, prob=1.0 / 11),
                eliminate_duplicates=True
            )
            res = minimize(problem, algorithm, ('n_gen', n_generations), seed=seed + idx, verbose=False)

            if res.X is not None:
                X_final = np.atleast_2d(res.X)
                raw_objs, raw_constrs = self.evaluator.evaluate_batch(X_final)
                for i in range(len(X_final)):
                    entry = self._create_result_entry(X_final[i], raw_objs[i], raw_constrs[i], cnt)
                    entry['weight_id'] = idx
                    self.results.append(entry)
                    cnt += 1

        self.execution_time = time.time() - start_time
        for r in self.results: r['time_seconds'] = self.execution_time
        return pd.DataFrame(self.results)


class GridSearchBaseline(BaselineMethod):
    def optimize(self, seed: int, **kwargs) -> pd.DataFrame:
        logger.info("Running Grid Search...")
        start_time = time.time()
        x_grid = []
        for sensor in [0.1, 0.5, 0.9]:
            for algo in [0.1, 0.5, 0.9]:
                for deploy in [0.0, 1.0]:
                    for cycle in np.linspace(0.1, 0.9, 5):
                        x = np.array([sensor, 0.5, 0.5, 0.5, algo, 0.7, 0.0, 0.5, deploy, 0.3, cycle])
                        x_grid.append(x)

        X = np.array(x_grid)
        objs, constrs = self.evaluator.evaluate_batch(X)
        self.execution_time = time.time() - start_time

        for i in range(len(X)):
            self.results.append(self._create_result_entry(X[i], objs[i], constrs[i], i))
        for r in self.results: r['time_seconds'] = self.execution_time
        return pd.DataFrame(self.results)


class ExpertHeuristicBaseline(BaselineMethod):
    def optimize(self, seed: int, **kwargs) -> pd.DataFrame:
        logger.info("Running Expert Heuristic...")
        start_time = time.time()
        configs = [
            np.array([0.95, 0.3, 0.5, 0.5, 0.8, 0.6, 0.0, 0.4, 1.0, 0.25, 0.2]),
            np.array([0.3, 0.7, 0.33, 0.33, 0.2, 0.8, 0.5, 0.8, 0.3, 0.5, 0.06]),
            np.array([0.88, 0.45, 0.5, 0.5, 0.65, 0.65, 0.0, 0.5, 0.8, 0.3, 0.15])
        ]
        X = np.array(configs)
        objs, constrs = self.evaluator.evaluate_batch(X)
        self.execution_time = time.time() - start_time

        for i in range(len(X)):
            self.results.append(self._create_result_entry(X[i], objs[i], constrs[i], i))
        for r in self.results: r['time_seconds'] = self.execution_time
        return pd.DataFrame(self.results)


class BaselineRunner:
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        from fitness_evaluation import EnhancedFitnessEvaluatorV3
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)

    def run_all_methods(self, mode='paper', seed=42, eval_budget=None) -> Dict[str, pd.DataFrame]:
        np.random.seed(seed)
        results = {}

        # 1. Random Search (Adjusts to budget in FAIR mode)
        rs = RandomSearchUniformBaseline(self.evaluator, self.config)
        if mode == 'fair' and eval_budget is not None:
            n_samples = eval_budget
        else:
            n_samples = self.config.n_random_samples  # Default 3000

        results['Random Search'] = rs.optimize(seed=seed, n_samples=n_samples)

        # 2. Weighted Sum (Usually fixed budget as per paper def, but keeps seed)
        ws = WeightedSumNSGA2Baseline(self.evaluator, self.config)
        results['Weighted Sum'] = ws.optimize(seed=seed)

        # 3. Grid
        gs = GridSearchBaseline(self.evaluator, self.config)
        results['Grid Search'] = gs.optimize(seed=seed)

        # 4. Expert
        ex = ExpertHeuristicBaseline(self.evaluator, self.config)
        results['Expert'] = ex.optimize(seed=seed)

        return results