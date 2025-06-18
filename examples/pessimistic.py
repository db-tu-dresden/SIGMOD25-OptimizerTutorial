from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable
from typing import Optional

import postbound as pb


class UpperBoundCardinalities(pb.CardinalityGenerator):
    def __init__(self, target_db: pb.Database) -> None:
        super().__init__(False)
        self.target_db = target_db

    def filtered_card(
        self, query: pb.SqlQuery, table: pb.TableReference
    ) -> pb.Cardinality:
        """Determines the (estimated) cardinality of a base relation after all filters have been applied"""
        subquery = pb.transform.extract_query_fragment(query, table)
        return self.target_db.optimizer().cardinality_estimate(subquery)

    # We provide two possible implementations:
    # once again using calculate_estimate(), or
    # using generate_plan_parameters(), which also shows how to determine the base joins and how to work with join orders

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

        lhs_card = self.filtered_card(query, lhs)
        rhs_card = self.filtered_card(query, rhs)

        upper_bound: pb.Cardinality = (
            min(lhs_card / lhs_mcf, rhs_card / rhs_mcf) * lhs_mcf * rhs_mcf
        )
        return upper_bound

    def generate_plan_parameters(
        self,
        query: pb.SqlQuery,
        join_order: Optional[pb.LogicalJoinTree],
        operator_assignment: Optional[pb.PhysicalOperatorAssignment],
    ) -> pb.PlanParameterization:
        #
        # We don't acutally need to implement this method
        # The CardinalityGenerator has a default implementation that performs the same workflow but delegates to
        # calculate_estimate() for the actual cardinality generation.
        # We implement it anyway to demonstrate how to work with the plan parameters and join orders
        #

        parameters = pb.PlanParameterization()

        if join_order:
            base_joins = [
                join.tables() for join in join_order.iterjoins() if join.is_base_join()
            ]
        else:
            base_joins = itertools.combinations(query.tables(), 2)

        for base_join in base_joins:
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

            lhs_card = self.filtered_card(query, lhs)
            rhs_card = self.filtered_card(query, rhs)

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

        # we can use the same strategy as in the BAO-light example
        assignment.set_operator_enabled_globally(pb.JoinOperator.NestedLoopJoin, False)
        assignment.set_operator_enabled_globally(pb.JoinOperator.HashJoin, True)
        assignment.set_operator_enabled_globally(pb.JoinOperator.SortMergeJoin, False)

        # alternatively, set the joins from the join order directly (make sure that the join order is actually set first)
        # for join in join_order.iterjoins():
        #    assignment.add(pb.JoinOperator.HashJoin, join)

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
        pb.MultiStageOptimizationPipeline(pg_instance)
        .setup_physical_operator_selection(PessimisticOperators())
        .setup_plan_parameterization(UpperBoundCardinalities(pg_instance))
        .build()
    )

    results = pb.optimize_and_execute_workload(workload, optimizer, logger="tqdm")
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
