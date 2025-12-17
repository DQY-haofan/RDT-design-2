#!/usr/bin/env python3
"""
配置管理器 - 文件夹结构增强版
自动创建 data/figures/validation 子目录，确保输出井井有条。
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ConfigManager:
    """中央配置管理"""

    # --- 基础文件路径 ---
    config_file: str = 'config.json'
    sensor_csv: str = 'sensors_data.txt'
    algorithm_csv: str = 'algorithms_data.txt'
    infrastructure_csv: str = 'infrastructure_data.txt'
    cost_benefit_csv: str = 'cost_benefit_data.txt'

    # --- 核心约束参数 (对齐论文) ---
    road_network_length_km: float = 500.0
    planning_horizon_years: int = 10
    budget_cap_usd: float = 10_000_000

    min_recall_threshold: float = 0.6
    max_latency_seconds: float = 500.0
    max_disruption_hours: float = 300.0
    max_energy_kwh_year: float = 200_000
    min_mtbf_hours: float = 1_500
    max_carbon_emissions_kgCO2e_year: float = 300_000

    # --- 运营参数 ---
    daily_wage_per_person: float = 1500
    fos_sensor_spacing_km: float = 0.1
    depreciation_rate: float = 0.1
    scenario_type: str = 'urban'
    carbon_intensity_factor: float = 0.417
    apply_seasonal_adjustments: bool = True
    traffic_volume_hourly: int = 2000
    default_lane_closure_ratio: float = 0.3

    # --- 高级字典参数 ---
    class_imbalance_penalties: Dict[str, float] = field(default_factory=lambda: {
        'Traditional': 0.05, 'ML': 0.02, 'DL': 0.01, 'PC': 0.03
    })

    network_quality_factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
        'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9},
        'mixed': {'Fiber': 0.9, '5G': 0.85, '4G': 0.95, 'LoRaWAN': 0.95}
    })

    redundancy_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Cloud': 10.0, 'OnPremise': 1.5, 'Edge': 2.0, 'Hybrid': 5.0
    })

    # --- 优化算法参数 ---
    n_objectives: int = 6
    n_partitions: int = 6
    population_size: int = 512
    n_generations: int = 200
    crossover_prob: float = 0.9
    crossover_eta: float = 20
    mutation_eta: float = 20
    random_seed: int = 42

    # --- 基线与并行 ---
    n_random_samples: int = 3000
    grid_resolution: int = 5
    weight_combinations: int = 50
    use_parallel: bool = True
    n_processes: int = -1

    # --- 输出与日志 ---
    output_dir: Path = field(default_factory=lambda: Path('./results'))
    # 定义子目录变量 (在 post_init 中初始化)
    log_dir: Path = field(default_factory=lambda: Path('./results/logs'))
    data_dir: Path = field(init=False)  # 存放 CSV
    figure_dir: Path = field(init=False)  # 存放 PNG/PDF
    validation_dir: Path = field(init=False)  # 存放 验证结果

    figure_format: List[str] = field(default_factory=lambda: ['png', 'pdf'])

    data_retention_years: int = 3
    enable_debug_output: bool = True

    def __post_init__(self):
        """初始化后处理：加载配置并创建目录结构"""
        # 1. 加载文件覆盖默认值
        if Path(self.config_file).exists():
            self.load_from_file(self.config_file)

        # 2. 处理计算属性
        if self.n_processes == -1:
            self.n_processes = max(1, mp.cpu_count() - 1)

        # 确保路径对象
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)

        # 3. 定义子目录结构
        self.data_dir = self.output_dir / 'data'
        self.figure_dir = self.output_dir / 'figures'
        self.validation_dir = self.output_dir / 'validation_results'

        # 4. 创建所有目录
        self._create_directories()

    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            field_names = {f.name for f in fields(self)}
            for key, value in data.items():
                if key in field_names:
                    if key in ['output_dir', 'log_dir']: value = Path(value)
                    setattr(self, key, value)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")

    def _create_directories(self):
        """创建完整的目录层级"""
        for p in [self.output_dir, self.log_dir, self.data_dir, self.figure_dir, self.validation_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def save_to_file(self, filepath: Optional[str] = None):
        filepath = filepath or self.config_file
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path): value = str(value)
                config_dict[key] = value
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_summary(self) -> str:
        return (f"配置摘要:\n"
                f"  输出目录: {self.output_dir}\n"
                f"  子目录: data/, figures/, logs/, validation_results/\n"
                f"  图表格式: {self.figure_format}\n"
                f"  种子: {self.random_seed}")