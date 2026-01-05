//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seqscan_as_indexscan.cpp
//
// Identification: src/optimizer/seqscan_as_indexscan.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <unordered_map>
#include "binder/tokens.h"
#include "common/macros.h"
#include "execution/expressions/abstract_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/logic_expression.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "optimizer/optimizer.h"

namespace bustub {

typedef std::unordered_map<uint32_t, std::vector<AbstractExpressionRef>> ColExprMap;

auto parseComparisonChild(AbstractExpressionRef left, AbstractExpressionRef right, ColExprMap &result) -> bool {
  auto left_col = dynamic_cast<ColumnValueExpression *>(left.get());
  auto right_const = dynamic_cast<ConstantValueExpression *>(right.get());
  if (!left_col || !right_const) {
    return false;
  }

  result[left_col->GetColIdx()].emplace_back(right);
  return true;
}

auto parseComparison(ComparisonExpression *expr, ColExprMap &result) -> bool {
  if (expr->comp_type_ != ComparisonType::Equal) {
    return false;
  }

  auto left = expr->GetChildAt(0);
  auto right = expr->GetChildAt(1);

  return parseComparisonChild(left, right, result) || parseComparisonChild(right, left, result);
}

auto parse(const AbstractExpressionRef &ref, ColExprMap &result) -> bool {
  auto comp = dynamic_cast<ComparisonExpression *>(ref.get());
  if (comp) {
    return parseComparison(comp, result);
  }

  auto expr = dynamic_cast<LogicExpression *>(ref.get());
  if (expr) {
    if (expr->logic_type_ == LogicType::Or) {
      return parse(expr->GetChildAt(0), result) && parse(expr->GetChildAt(1), result);
    }
  }
  return false;
}

/**
 * @brief Optimizes seq scan as index scan if there's an index on a table
 */
auto Optimizer::OptimizeSeqScanAsIndexScan(const bustub::AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeSeqScanAsIndexScan(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));

  if (plan->GetType() != PlanType::SeqScan) {
    return optimized_plan;
  }

  const auto &seq_scan_plan = dynamic_cast<const SeqScanPlanNode &>(*plan);
  if (!seq_scan_plan.filter_predicate_) {
    return optimized_plan;
  }

  ColExprMap result;
  if (!parse(seq_scan_plan.filter_predicate_, result) || result.size() != 1) {
    return optimized_plan;
  }

  const auto table_info = catalog_.GetTable(seq_scan_plan.GetTableOid());
  const auto indices = catalog_.GetTableIndexes(table_info->name_);

  for (auto &index : indices) {
    auto attr = index->index_->GetKeyAttrs();
    BUSTUB_ASSERT(attr.size() >= 1, "Index must have at least one key attribute");

    if (attr.size() > 1) {
      continue;
    }

    if (attr[0] == result.begin()->first) {
      return std::make_shared<IndexScanPlanNode>(seq_scan_plan.output_schema_, seq_scan_plan.GetTableOid(),
                                                 index->index_oid_, nullptr, result.begin()->second);
    }
  }

  return optimized_plan;
}

}  // namespace bustub
