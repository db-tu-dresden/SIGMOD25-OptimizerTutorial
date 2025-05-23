from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd
import postbound as pb
import sentence_transformers as st
from sklearn.ensemble import GradientBoostingRegressor


class MSCNlight(pb.CardinalityGenerator):
    def __init__(self) -> None:
        super().__init__(False)
        self.model = GradientBoostingRegressor()
        self.featurizer = st.SentenceTransformer("all-MiniLM-L6-v2")

    def train(self, samples: pd.DataFrame) -> None:
        samples["features"] = samples["query"].map(self.featurizer.encode)
        self.model.fit(np.stack(samples["features"]), samples["cardinality"])

    def calculate_estimate(
        self, query: pb.SqlQuery, tables: Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_query_fragment(query, tables)
        features = self.featurizer.encode(str(subquery))
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
        pb.TextBookOptimizationPipeline(pg_instance)
        .setup_cardinality_estimator(mscn)
        .build()
    )

    results = pb.optimize_and_execute_workload(workload, optimizer)
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
