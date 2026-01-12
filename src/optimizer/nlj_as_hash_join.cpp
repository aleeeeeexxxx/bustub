//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nlj_as_hash_join.cpp
//
// Identification: src/optimizer/nlj_as_hash_join.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <memory>
#include <vector>
#include "catalog/column.h"
#include "catalog/schema.h"
#include "common/exception.h"
#include "common/macros.h"
#include "execution/expressions/abstract_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/logic_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/filter_plan.h"
#include "execution/plans/hash_join_plan.h"
#include "execution/plans/nested_loop_join_plan.h"
#include "execution/plans/projection_plan.h"
#include "optimizer/optimizer.h"
#include "type/type_id.h"

namespace bustub {

struct JoinExprs {
  std::vector<AbstractExpressionRef> left_exprs_;
  std::vector<AbstractExpressionRef> right_exprs_;

  JoinExprs() = default;

  auto Add(uint32_t index, AbstractExpressionRef expr) -> void {
    if (index == 0) {
      left_exprs_.push_back(expr);
    } else {
      right_exprs_.push_back(expr);
    }
  }
};

auto parse(const AbstractExpressionRef &plan, JoinExprs &exprs) -> bool {
  auto comp_expr = dynamic_cast<const ComparisonExpression *>(plan.get());
  if (comp_expr != nullptr) {
    if (comp_expr->comp_type_ == ComparisonType::Equal) {
      auto left_col_expr = dynamic_cast<const ColumnValueExpression *>(comp_expr->GetChildAt(0).get());
      auto right_col_expr = dynamic_cast<const ColumnValueExpression *>(comp_expr->GetChildAt(1).get());

      if (left_col_expr != nullptr && right_col_expr != nullptr) {
        if (left_col_expr->GetTupleIdx() == right_col_expr->GetTupleIdx()) {
          return false;
        }

        exprs.Add(left_col_expr->GetTupleIdx(), comp_expr->GetChildAt(0));
        exprs.Add(right_col_expr->GetTupleIdx(), comp_expr->GetChildAt(1));
        return true;
      }
    }
  }

  auto logic_expr = dynamic_cast<const LogicExpression *>(plan.get());
  if (logic_expr != nullptr) {
    if (logic_expr->logic_type_ == LogicType::And) {
      return parse(logic_expr->GetChildAt(0), exprs) && parse(logic_expr->GetChildAt(1), exprs);
    }
  }

  return false;
}

/**
 * @brief optimize nested loop join into hash join.
 * In the starter code, we will check NLJs with exactly one equal condition. You can further support optimizing joins
 * with multiple eq conditions.
 */
auto Optimizer::OptimizeNLJAsHashJoin(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  // Note for Spring 2025: You should support join keys of any number of conjunction of equi-conditions:
  // E.g. <column expr> = <column expr> AND <column expr> = <column expr> AND ...

  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeNLJAsHashJoin(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));

  if (optimized_plan->GetType() == PlanType::NestedLoopJoin) {
    const auto &nlj_plan = dynamic_cast<const NestedLoopJoinPlanNode &>(*optimized_plan);

    JoinExprs exprs;

    if (parse(nlj_plan.Predicate(), exprs)) {
      auto hash_join_plan =
          std::make_shared<HashJoinPlanNode>(nlj_plan.output_schema_, nlj_plan.GetLeftPlan(), nlj_plan.GetRightPlan(),
                                             exprs.left_exprs_, exprs.right_exprs_, nlj_plan.GetJoinType());
      return hash_join_plan;
    }
  }

  return optimized_plan;
}

}  // namespace bustub
