import argparse
import itertools
from collections.abc import Iterable
from typing import Optional

import postbound as pb


class UpperBoundCardinalities(pb.CardinalityGenerator):
    def __init__(self, target_db: pb.Database) -> None:
        self.target_db = target_db

    # Two alternatives
    # once again using calculate_estimate(), or
    # using generate_plan_parameters(), which also demonstrates how to determine the base joins

    def calculate_estimate(
        self, query: pb.SqlQuery, tables: Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        if len(tables) != 2:
            return pb.Cardinality.unknown()

        lhs, rhs = list(tables)
        join_predicate = query.predicates().joins_between(lhs, rhs)
        if not join_predicate:
            return pb.Cardinality.unknown()

        lhs_column: pb.ColumnReference = pb.util.simplify(
            join_predicate.columns_of(lhs)
        )
        rhs_column: pb.ColumnReference = pb.util.simplify(
            join_predicate.columns_of(rhs)
        )

        lhs_topk = self.target_db.statistics().most_common_values(lhs_column, k=1)
        lhs_mcf: int = lhs_topk[1]

        rhs_topk = self.target_db.statistics().most_common_values(rhs_column, k=1)
        rhs_mcf: int = rhs_topk[1]

        lhs_card: int = self.target_db.statistics().total_rows(lhs)
        rhs_card: int = self.target_db.statistics().total_rows(rhs)

        upper_bound = min(lhs_card / lhs_mcf, rhs_card / rhs_mcf) * lhs_mcf * rhs_mcf
        return pb.Cardinality(upper_bound)

    def generate_plan_parameters(
        self,
        query: pb.SqlQuery,
        join_order: Optional[pb.LogicalJoinTree],
        operator_assignment: Optional[pb.PhysicalOperatorAssignment],
    ) -> pb.PlanParameterization:
        parameters = pb.PlanParameterization()

        for base_join in itertools.combinations(query.tables(), 2):
            lhs, rhs = base_join
            join_predicate = query.predicates().joins_between(lhs, rhs)
            if not join_predicate:
                continue

            lhs_column: pb.ColumnReference = pb.util.simplify(
                join_predicate.columns_of(lhs)
            )
            rhs_column: pb.ColumnReference = pb.util.simplify(
                join_predicate.columns_of(rhs)
            )

            lhs_topk = self.target_db.statistics().most_common_values(lhs_column, k=1)
            lhs_mcf: int = lhs_topk[1]

            rhs_topk = self.target_db.statistics().most_common_values(rhs_column, k=1)
            rhs_mcf: int = rhs_topk[1]

            lhs_card: int = self.target_db.statistics().total_rows(lhs)
            rhs_card: int = self.target_db.statistics().total_rows(rhs)

            upper_bound = (
                min(lhs_card / lhs_mcf, rhs_card / rhs_mcf) * lhs_mcf * rhs_mcf
            )
            parameters.add_cardinality_hint(base_join, pb.Cardinality(upper_bound))

        return parameters

    def describe(self) -> pb.util.jsondict:
        return {"name": "UpperBound-cardinalities"}


class PessimisticOperators(pb.PhysicalOperatorSelection):
    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: Optional[pb.LogicalJoinTree]
    ) -> pb.PhysicalOperatorAssignment:
        assignment = pb.PhysicalOperatorAssignment()
        assignment.set_operator_enabled_globally(pb.JoinOperator.NestedLoopJoin, False)
        assignment.set_operator_enabled_globally(pb.JoinOperator.HashJoin, True)
        assignment.set_operator_enabled_globally(pb.JoinOperator.SortMergeJoin, False)
        return assignment

    def describe(self) -> pb.util.jsondict:
        return {"name": "Pessimistic-operators"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upper bound cardinality estimator with pessimistic operator selection"
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

    optimizer = (
        pb.TwoStageOptimizationPipeline(pg_instance)
        .setup_physical_operator_selection(PessimisticOperators())
        .setup_plan_parameterization(UpperBoundCardinalities(pg_instance))
        .build()
    )

    results = pb.optimize_and_execute_workload(workload.first(5), optimizer)
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
