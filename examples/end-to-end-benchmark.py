from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import warnings

import pandas as pd
import postbound as pb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

Pipelines = {"native", "mscn-l", "bao-l", "pess", "all"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark of the demo optimizers"
    )
    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        required=True,
        help="Config file for the Postgres connection",
    )
    parser.add_argument(
        "--workload",
        "-w",
        type=str,
        required=True,
        help="Directory containing the evaluation queries",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=Pipelines,
        default=["all"],
        help="The pipelines to evaluate",
    )
    parser.add_argument(
        "--samples-cards",
        type=pathlib.Path,
        default=pathlib.Path("data/cardinality-samples.csv"),
        help="CSV file containing the cardinality training data",
    )
    parser.add_argument(
        "--samples-qeps",
        type=pathlib.Path,
        default=pathlib.Path("data/qep-samples.csv"),
        help="CSV file containing the query plan training data",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory to save the result CSVs",
    )

    args = parser.parse_args()
    pg_instance = pb.postgres.connect(config_file=args.connect)
    workload = pb.workloads.read_workload(args.workload)
    logger = pb.util.make_logger(prefix=pb.util.timestamp)
    selected_models = set(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_all = "all" in selected_models
    query_prep = pb.experiments.QueryPreparationService(prewarm=True, analyze=True)

    if benchmark_all or "native" in selected_models:
        logger("Benchmarking baseline PG optimizer")
        results = pb.execute_workload(
            workload,
            pg_instance,
            query_preparation=query_prep,
            timeout=120,
            include_labels=True,
            logger="tqdm",
        )
        results = pb.experiments.prepare_export(results)
        results.to_csv(args.out_dir / "benchmark-native.csv", index=False)

    if benchmark_all or "mscn-l" in selected_models:
        mscn = importlib.import_module("examples.mscn-light")
        mscn_model = mscn.MSCNlight()
        train_data = pd.read_csv(args.samples_cards)

        logger("Training MSCN-light model")
        mscn_model.train(train_data)

        logger("Benchmarking MSCN-light")
        optimizer = (
            pb.MultiStageOptimizationPipeline(pg_instance)
            .setup_plan_parameterization(mscn_model)
            .build()
        )
        results = pb.optimize_and_execute_workload(
            workload,
            optimizer,
            query_preparation=query_prep,
            timeout=120,
            include_labels=True,
            logger="tqdm",
        )
        results = pb.experiments.prepare_export(results)
        results.to_csv(args.out_dir / "benchmark-mscn-light.csv", index=False)

    if benchmark_all or "bao-l" in selected_models:
        bao = importlib.import_module("examples.bao-light")
        bao_model = bao.BAOlight(pg_instance)
        train_data = pd.read_csv(args.samples_qeps)

        logger("Training BAO-light model")
        bao_model.train(train_data)

        logger("Benchmarking BAO-light")
        optimizer = (
            pb.IntegratedOptimizationPipeline(pg_instance)
            .setup_optimization_algorithm(bao_model)
            .build()
        )
        results = pb.optimize_and_execute_workload(
            workload,
            optimizer,
            query_preparation=query_prep,
            timeout=120,
            include_labels=True,
            logger="tqdm",
        )
        results = pb.experiments.prepare_export(results)
        results.to_csv(args.out_dir / "benchmark-bao-light.csv", index=False)

    if benchmark_all or "pess" in selected_models:
        pess = importlib.import_module("examples.pessimistic")
        logger("Benchmarking pessimnistic otimizer")
        upper_bound_est = pess.UpperBoundCardinalities(pg_instance)
        pess_operators = pess.PessimisticOperators()
        optimizer = (
            pb.MultiStageOptimizationPipeline(pg_instance)
            .setup_physical_operator_selection(pess_operators)
            .setup_plan_parameterization(upper_bound_est)
            .build()
        )
        results = pb.optimize_and_execute_workload(
            workload,
            optimizer,
            query_preparation=query_prep,
            timeout=120,
            include_labels=True,
            logger="tqdm",
        )
        results = pb.experiments.prepare_export(results)
        results.to_csv(args.out_dir / "benchmark-pessimistic.csv", index=False)


if __name__ == "__main__":
    main()
