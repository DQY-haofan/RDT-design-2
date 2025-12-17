#!/usr/bin/env python3
"""
fitness_evaluation.py (原 evaluation.py)
重命名以避免与 python 'evaluation' 库冲突
增强的适应度评估模块 V3 - 专家级改进版
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from rdflib.namespace import RDF
from rdflib import Graph, Namespace

logger = logging.getLogger(__name__)

RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class SolutionMapper:
    """解决方案映射器"""

    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        self._decode_cache = {}

    def _cache_components(self):
        """缓存所有可用组件"""
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []

        logger.info("缓存本体组件...")

        sensor_patterns = ['Sensor', 'sensor', 'LiDAR', 'Camera', 'Scanner']
        algo_patterns = ['Algorithm', 'algorithm']
        deploy_patterns = ['Deployment', 'Compute', 'Edge', 'Cloud']

        for s, p, o in self.g:
            if p == RDF.type and str(s).startswith('http://example.org/rmtwin#'):
                subject_str = str(s)
                type_str = str(o)

                if any(p in type_str for p in sensor_patterns):
                    if subject_str not in self.sensors: self.sensors.append(subject_str)
                elif any(p in type_str for p in algo_patterns):
                    if subject_str not in self.algorithms: self.algorithms.append(subject_str)
                elif 'Storage' in type_str:
                    if subject_str not in self.storage_systems: self.storage_systems.append(subject_str)
                elif 'Communication' in type_str:
                    if subject_str not in self.comm_systems: self.comm_systems.append(subject_str)
                elif any(p in type_str for p in deploy_patterns):
                    if subject_str not in self.deployments: self.deployments.append(subject_str)

        # 默认值兜底
        if not self.sensors: self.sensors = ["http://example.org/rmtwin#IoT_LoRaWAN_Sensor"]
        if not self.algorithms: self.algorithms = ["http://example.org/rmtwin#Traditional_Canny_Optimized"]
        if not self.storage_systems: self.storage_systems = ["http://example.org/rmtwin#Storage_AWS_S3_Standard"]
        if not self.comm_systems: self.comm_systems = ["http://example.org/rmtwin#Communication_LoRaWAN_Gateway"]
        if not self.deployments: self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_GPU_A4000"]

    def decode_solution(self, x: np.ndarray) -> Dict:
        """解码解决方案向量"""
        x_key = tuple(float(xi) for xi in x)
        if x_key in self._decode_cache: return self._decode_cache[x_key]

        # 确保索引在有效范围内
        def get_comp(lst, val):
            return lst[int(val * len(lst)) % len(lst)]

        config = {
            'sensor': get_comp(self.sensors, x[0]),
            'data_rate': 10 + x[1] * 90,
            'geo_lod': ['Micro', 'Meso', 'Macro'][int(x[2] * 3) % 3],
            'cond_lod': ['Micro', 'Meso', 'Macro'][int(x[3] * 3) % 3],
            'algorithm': get_comp(self.algorithms, x[4]),
            'detection_threshold': 0.1 + x[5] * 0.8,
            'storage': get_comp(self.storage_systems, x[6]),
            'communication': get_comp(self.comm_systems, x[7]),
            'deployment': get_comp(self.deployments, x[8]),
            'crew_size': int(1 + x[9] * 9),
            'inspection_cycle': int(1 + x[10] * 364)
        }

        self._decode_cache[x_key] = config
        return config


class EnhancedFitnessEvaluatorV3:
    """增强的适应度评估器 V3"""

    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph)
        self._property_cache = {}
        self._initialize_cache()
        self.depreciation_rates = {'MMS': 0.15, 'UAV': 0.20, 'TLS': 0.12, 'Handheld': 0.12, 'Vehicle': 0.15,
                                   'IoT': 0.10}

    def _initialize_cache(self):
        """预缓存属性查询，大幅提升速度"""
        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasAnnualOpCostUSD',
            'hasEnergyConsumptionW', 'hasMTBFHours', 'hasCoverageEfficiencyKmPerDay',
            'hasRecall', 'hasPrecision', 'hasDataVolumeGBPerKm', 'hasOperatingSpeedKmh'
        ]

        components = (self.solution_mapper.sensors + self.solution_mapper.algorithms +
                      self.solution_mapper.storage_systems + self.solution_mapper.comm_systems +
                      self.solution_mapper.deployments)

        for comp in components:
            self._property_cache[comp] = {}
            for prop in properties:
                # 简单查询
                q = f"""PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
                        SELECT ?v WHERE {{ <{comp}> rdtco:{prop} ?v }}"""
                for row in self.g.query(q):
                    try:
                        self._property_cache[comp][prop] = float(row[0])
                    except:
                        self._property_cache[comp][prop] = str(row[0])

    def _query_property(self, subject, prop, default=0.0):
        if subject in self._property_cache and prop in self._property_cache[subject]:
            return self._property_cache[subject][prop]
        return default

    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量评估"""
        n = len(X)
        objs = np.zeros((n, 6))
        constrs = np.zeros((n, 5))

        for i in range(n):
            objs[i], constrs[i] = self._evaluate_single(X[i])

        return objs, constrs

    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        config = self.solution_mapper.decode_solution(x)

        # 1. 成本 (Cost)
        f1 = self._calc_cost(config)

        # 2. 召回率 (1 - Recall)
        recall_score = self._calc_recall(config)
        f2 = 1.0 - recall_score

        # 3. 延迟 (Latency)
        f3 = self._calc_latency(config)

        # 4. 干扰 (Disruption)
        f4 = self._calc_disruption(config)

        # 5. 碳排放 (Carbon)
        f5 = self._calc_carbon(config)

        # 6. 可靠性 (1 / MTBF)
        mtbf = self._calc_reliability(config)
        f6 = 1.0 / mtbf if mtbf > 0 else 1.0

        # 约束 (g <= 0 为可行)
        g1 = f3 - self.config.max_latency_seconds
        g2 = self.config.min_recall_threshold - recall_score
        g3 = f1 - self.config.budget_cap_usd
        g4 = f5 - self.config.max_carbon_emissions_kgCO2e_year
        g5 = self.config.min_mtbf_hours - mtbf

        return np.array([f1, f2, f3, f4, f5, f6]), np.array([g1, g2, g3, g4, g5])

    # --- 简化的计算逻辑 (核心逻辑保持与原 Evaluation 模块一致) ---
    def _calc_cost(self, c):
        # 资本支出 + 运营支出
        init_cost = self._query_property(c['sensor'], 'hasInitialCostUSD')
        op_cost = self._query_property(c['sensor'], 'hasOperationalCostUSDPerDay')
        cycle_days = c['inspection_cycle']
        daily_wage = self.config.daily_wage_per_person
        crew_cost = c['crew_size'] * daily_wage * (365 / cycle_days)

        # 简单折旧模型
        annual = (init_cost * 0.2) + (op_cost * 365) + crew_cost
        return annual * self.config.planning_horizon_years

    def _calc_recall(self, c):
        base = self._query_property(c['algorithm'], 'hasRecall', 0.7)
        # LOD 影响
        lod_factor = 1.1 if c['geo_lod'] == 'Micro' else 0.9
        return min(0.99, base * lod_factor)

    def _calc_latency(self, c):
        # 数据量 * 传输 + 处理
        vol = self._query_property(c['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        # 假设 500km 路网
        total_vol = vol * self.config.road_network_length_km
        bandwidth = 0.1  # GB/s simplified
        return (total_vol / bandwidth) + 60  # + overhead

    def _calc_disruption(self, c):
        speed = self._query_property(c['sensor'], 'hasOperatingSpeedKmh', 0)
        if speed > 50: return 10.0  # 移动扫描，干扰小
        return 200.0  # 需要封路

    def _calc_carbon(self, c):
        watts = self._query_property(c['sensor'], 'hasEnergyConsumptionW', 100)
        kwh = (watts * 24 * 365) / 1000
        return kwh * self.config.carbon_intensity_factor

    def _calc_reliability(self, c):
        return self._query_property(c['sensor'], 'hasMTBFHours', 5000)