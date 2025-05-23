from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd
import postbound as pb
from postbound.experiments import querygen


@dataclass
class TrainingSample:
    query: str
    query_plan: str
    runtime: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        required=True,
        help="Config file for the Postgres connection",
    )
    parser.add_argument(
        "--n-queries",
        "-n",
        type=int,
        default=1000,
        help="Number of queries to generate",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        required=True,
        help="Output CSV file for the training samples",
    )

    args = parser.parse_args()
    pg_instance = pb.postgres.connect(config_file=args.connect)
    logger = pb.util.make_logger(prefix=pb.util.timestamp)

    tried_queries: set[pb.SqlQuery] = set()
    generated_queries: int = 0
    query_generator = querygen.generate_query(
        pg_instance,
        count_star=True,
        min_tables=2,
        max_tables=5,
        min_filters=1,
        max_filters=4,
        filter_key_columns=False,
        numeric_filters=True,
    )

    # We "stream" the generated queries to the CSV file. Therefore, we first need to create the CSV skeleton with the header
    # column. Afterwards, we can create mini dataframes of one query each and append them to the CSV file.

    df_header = pd.DataFrame([], columns=["query", "query_plan", "runtime"])
    df_header.to_csv(args.out, index=False, mode="w")

    while generated_queries < args.n_queries:
        if generated_queries and generated_queries % (args.n_queries // 10) == 0:
            logger(
                "Accepted",
                generated_queries,
                "queries, tried",
                len(tried_queries),
                "queries",
            )

        query = next(query_generator)
        if query in tried_queries:
            continue
        tried_queries.add(query)

        query_plan = pg_instance.optimizer().query_plan(query)
        start_time = time.monotonic_ns()
        result_set = pg_instance.execute_with_timeout(
            query,
            timeout=120,
        )
        end_time = time.monotonic_ns()
        timeout = result_set is None
        if timeout:
            continue

        runtime = (end_time - start_time) / 1_000_000_000  # nanoseconds to seconds

        sample = TrainingSample(
            query=pb.qal.format_quick(query),
            query_plan=query_plan.inspect(),
            runtime=runtime,
        )

        df = pd.DataFrame([sample])
        df.to_csv(args.out, index=False, mode="a", header=False)

        generated_queries += 1


if __name__ == "__main__":
    main()
