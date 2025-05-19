import argparse

import numpy as np
import pandas as pd
import postbound as pb
import sentence_transformers as st
from sklearn.ensemble import GradientBoostingRegressor


class BAOlight(pb.CompleteOptimizationAlgorithm):
    def __init__(self, target_db: pb.Database) -> None:
        self.target_db = target_db
        self.model = GradientBoostingRegressor()
        self.featurizer = st.SentenceTransformer("all-MiniLM-L6-v2")

        no_nlj = pb.PhysicalOperatorAssignment()
        no_nlj.set_operator_enabled_globally(pb.JoinOperator.NestedLoopJoin, False)

        no_hash = pb.PhysicalOperatorAssignment()
        no_hash.set_operator_enabled_globally(pb.JoinOperator.HashJoin, False)

        no_merge = pb.PhysicalOperatorAssignment()
        no_merge.set_operator_enabled_globally(pb.JoinOperator.SortMergeJoin, False)

        self.hint_sets = [no_nlj, no_hash, no_merge]

    def train(self, samples: pd.DataFrame) -> None:
        samples["features"] = samples["query_plan"].map(self.featurizer.encode)
        self.model.fit(np.stack(samples["features"]), samples["runtime"])

    def optimize_query(self, query: pb.SqlQuery) -> pb.QueryPlan:
        predictions: dict[pb.QueryPlan, np.float64] = {}

        for hint_set in self.hint_sets:
            hinted_query = self.target_db.hinting().generate_hints(
                query, physical_operators=hint_set
            )
            query_plan = self.target_db.optimizer().query_plan(hinted_query)

            prediction = self.model.predict(
                [self.featurizer.encode(query_plan.inspect())]
            )
            predictions[query_plan] = prediction[0]

        return pb.util.argmin(predictions)

    def describe(self) -> pb.util.jsondict:
        return {
            "name": "BAO-light",
            "target_db": self.target_db.describe(),
            "hint_sets": self.hint_sets,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="BAO-style optimizer hint selection")
    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        required=True,
        help="CSV file containing the training data",
    )
    parser.add_argument(
        "--workload",
        "-w",
        type=str,
        required=True,
        help="Directory containing the evaluation queries",
    )
    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        required=True,
        help="Config file for the Postgres connection",
    )
    parser.add_argument(
        "--out", "-o", type=str, required=True, help="Output CSV file for the results"
    )

    args = parser.parse_args()
    pg_instance = pb.postgres.connect(config_file=args.connect)
    workload = pb.workloads.read_workload(args.workload)
    samples = pd.read_csv(args.samples)

    bao = BAOlight(pg_instance)
    bao.train(samples)

    optimizer = (
        pb.IntegratedOptimizationPipeline(pg_instance)
        .setup_optimization_algorithm(bao)
        .build()
    )

    results = pb.optimize_and_execute_workload(workload.first(5), optimizer)
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
