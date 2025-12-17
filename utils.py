#!/usr/bin/env python3
"""
Utility Functions for RMTwin Optimization
Includes robust metric calculations and helper functions.
"""
import logging
import json
from typing import Dict

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(debug=False, log_dir=Path('./results/logs')):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_dir / 'run.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def filter_nondominated(F: np.ndarray) -> np.ndarray:
    """
    Filter a set of points to only return the non-dominated set.
    Robust implementation handling different pymoo versions and edge cases.
    """
    if len(F) == 0: return F

    try:
        # Prefer pymoo's efficient implementation
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting()
        fronts = nds.do(F, only_non_dominated_front=True)

        # FIX: pymoo often returns a numpy array of indices for the first front
        # We need to use these indices to slice F
        return F[fronts]

    except Exception as e:
        # Robust Fallback: O(N^2) pairwise comparison
        # Suitable for small to medium sets (N < 5000)
        is_dominated = np.zeros(len(F), dtype=bool)
        for i in range(len(F)):
            if is_dominated[i]: continue
            for j in range(len(F)):
                if i == j: continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return F[~is_dominated]


def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Calculate Hypervolume of F relative to ref_point.
    Assumes F is minimized and normalized.
    """
    if len(F) == 0: return 0.0
    try:
        from pymoo.indicators.hv import HV
        # Filter out points that exceed the reference point (they contribute 0)
        mask = np.all(F <= ref_point, axis=1)
        F_valid = F[mask]

        if len(F_valid) == 0: return 0.0

        ind = HV(ref_point=ref_point)
        return ind(F_valid)
    except Exception:
        return 0.0


def calculate_a12_effect_size(list1, list2):
    """Vargha and Delaney's A12 effect size."""
    if len(list1) == 0 or len(list2) == 0: return 0.5
    more = 0
    same = 0
    for x in list1:
        for y in list2:
            if x > y:
                more += 1
            elif x == y:
                same += 1
    return (more + 0.5 * same) / (len(list1) * len(list2))


def save_results_summary(results: Dict, config):
    """Save high-level summary JSON."""
    path = config.output_dir / 'summary_report.json'

    def convert(o):
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, pd.DataFrame): return o.to_dict(orient='records')
        return str(o)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config),
        "keys": list(results.keys())
    }

    with open(path, 'w') as f:
        json.dump(summary, f, default=convert, indent=2)
    logger.info(f"Summary saved to {path}")