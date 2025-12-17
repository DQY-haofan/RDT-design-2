#!/usr/bin/env python3
"""
RMTwin Optimization Framework - Main Entry Point (Folder Organized)
"""
import argparse
import logging
import numpy as np
import random
from config_manager import ConfigManager
from ontology_manager import OntologyManager
from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from validation_harness import StatisticalValidator
from visualization import Visualizer
from utils import setup_logging, save_results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--run-validation', action='store_true')
    parser.add_argument('--seeds', default="42,43,44,45,46")
    parser.add_argument('--baseline-mode', default='fair', choices=['paper', 'fair'])
    parser.add_argument('--skip-optimization', action='store_true')
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--skip-visualization', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = ConfigManager(args.config)

    if args.run_validation:
        logger = setup_logging(debug=args.debug, log_dir=config.log_dir)
        logger.info(">>> STATISTICAL VALIDATION MODE <<<")

        ontology_manager = OntologyManager()
        ontology_graph = ontology_manager.populate_from_csv_files(
            config.sensor_csv, config.algorithm_csv,
            config.infrastructure_csv, config.cost_benefit_csv
        )

        seeds = [int(s) for s in args.seeds.split(',')]
        validator = StatisticalValidator(ontology_graph, config, seeds)
        validator.run_validation(baseline_mode=args.baseline_mode)

    else:
        # Standard Single Run
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        logger = setup_logging(debug=args.debug, log_dir=config.log_dir)
        logger.info(">>> SINGLE EXPERIMENT MODE <<<")
        logger.info(config.get_summary())

        try:
            ontology_manager = OntologyManager()
            ontology_graph = ontology_manager.populate_from_csv_files(
                config.sensor_csv, config.algorithm_csv,
                config.infrastructure_csv, config.cost_benefit_csv
            )

            all_results = {}

            if not args.skip_optimization:
                optimizer = RMTwinOptimizer(ontology_graph, config)
                pareto_df, final_pop_df, history = optimizer.optimize()

                all_results['nsga3'] = {'dataframe': pareto_df, 'history': history}

                # 保存 CSV 到 data 子目录
                if not pareto_df.empty:
                    pareto_df.to_csv(config.data_dir / 'pareto_solutions_6obj_fixed.csv', index=False)
                if not final_pop_df.empty:
                    final_pop_df.to_csv(config.data_dir / 'final_population.csv', index=False)

            if not args.skip_baselines:
                runner = BaselineRunner(ontology_graph, config)
                bl_results = runner.run_all_methods(mode='paper', seed=config.random_seed)
                all_results['baselines'] = {'dataframes': bl_results}

                # 保存 CSV 到 data 子目录
                for k, v in bl_results.items():
                    v.to_csv(config.data_dir / f'baseline_{k.replace(" ", "_")}.csv', index=False)

            if not args.skip_visualization:
                try:
                    viz = Visualizer(config)
                    nsga3_data = all_results.get('nsga3', {}).get('dataframe')
                    baseline_data = all_results.get('baselines', {}).get('dataframes')
                    history_data = all_results.get('nsga3', {}).get('history')

                    if nsga3_data is not None and not nsga3_data.empty:
                        viz.create_all_figures(nsga3_data, baseline_data, history_data)
                except Exception as e:
                    logger.warning(f"Viz error: {e}")

            save_results_summary(all_results, config)

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    main()