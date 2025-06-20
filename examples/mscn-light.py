from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import postbound as pb
import sentence_transformers as st
from sklearn.ensemble import GradientBoostingRegressor

# We suppress hint warnings here, because pg_hint_plan cannot hint cardinalities for base tables.
# PostBOUND issues warnings if it encounters such hints for pg_hint_plan.
warnings.filterwarnings("ignore", category=pb.db.HintWarning, module="postbound")


class MSCNlight(pb.CardinalityGenerator):
    def __init__(self) -> None:
        super().__init__(False)
        self.model = GradientBoostingRegressor()
        self.embedding = st.SentenceTransformer("all-MiniLM-L6-v2")

    def featurize(self, query: pb.SqlQuery) -> np.ndarray:
        tables = query.tables()
        predicates = query.predicates()

        from_clause_features = self.embedding.encode([str(tables)])
        join_features = self.embedding.encode([str(predicates.joins())])
        filter_features = self.embedding.encode([str(predicates.filters())])

        return np.concat([from_clause_features, join_features, filter_features], axis=1)[0]

    def train(self, samples: pd.DataFrame) -> None:
        samples["features"] = samples["query"].map(pb.parse_query).map(self.featurize)
        self.model.fit(np.stack(samples["features"]), samples["cardinality"])

    def calculate_estimate(
        self, query: pb.SqlQuery, tables: Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_query_fragment(query, tables)
        features = self.featurize(subquery)
        estimate: np.float64 = self.model.predict([features])[0]
        return pb.Cardinality(max(estimate, 0))

    def describe(self) -> pb.util.jsondict:
        return {"name": "MSCN-light"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSCN-style supervised cardinality estimator"
    )
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

    mscn = MSCNlight()
    mscn.train(samples)

    optimizer = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .setup_plan_parameterization(mscn)
        .build()
    )

    results = pb.optimize_and_execute_workload(workload, optimizer, logger="tqdm")
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
