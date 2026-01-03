// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Physical Optimizer Rules

use std::sync::Arc;

use super::TakeExec;
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DFResult,
    physical_optimizer::{optimizer::PhysicalOptimizer, PhysicalOptimizerRule},
    physical_plan::{
        coalesce_batches::CoalesceBatchesExec, projection::ProjectionExec, ExecutionPlan,
    },
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{expressions::Column, PhysicalExpr};

/// Rule that eliminates [TakeExec] nodes that are immediately followed by another [TakeExec].
#[derive(Debug)]
pub struct CoalesceTake;

impl CoalesceTake {
    fn field_order_differs(old_schema: &ArrowSchema, new_schema: &ArrowSchema) -> bool {
        old_schema
            .fields
            .iter()
            .zip(&new_schema.fields)
            .any(|(old, new)| old.name() != new.name())
    }

    fn remap_collapsed_output(
        old_schema: &ArrowSchema,
        new_schema: &ArrowSchema,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let mut project_exprs = Vec::with_capacity(old_schema.fields.len());
        for field in &old_schema.fields {
            project_exprs.push((
                Arc::new(Column::new_with_schema(field.name(), new_schema).unwrap())
                    as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        Arc::new(ProjectionExec::try_new(project_exprs, plan).unwrap())
    }

    fn collapse_takes(
        inner_take: &TakeExec,
        outer_take: &TakeExec,
        outer_exec: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let inner_take_input = inner_take.children()[0].clone();
        let old_output_schema = outer_take.schema();
        let collapsed = outer_exec
            .with_new_children(vec![inner_take_input])
            .unwrap();
        let new_output_schema = collapsed.schema();

        // It's possible that collapsing the take can change the field order.  This disturbs DF's planner and
        // so we must restore it.
        if Self::field_order_differs(&old_output_schema, &new_output_schema) {
            Self::remap_collapsed_output(&old_output_schema, &new_output_schema, collapsed)
        } else {
            collapsed
        }
    }
}

impl PhysicalOptimizerRule for CoalesceTake {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                if let Some(outer_take) = plan.as_any().downcast_ref::<TakeExec>() {
                    let child = outer_take.children()[0];
                    // Case 1: TakeExec -> TakeExec
                    if let Some(inner_take) = child.as_any().downcast_ref::<TakeExec>() {
                        return Ok(Transformed::yes(Self::collapse_takes(
                            inner_take,
                            outer_take,
                            plan.clone(),
                        )));
                    // Case 2: TakeExec -> CoalesceBatchesExec -> TakeExec
                    } else if let Some(exec_child) =
                        child.as_any().downcast_ref::<CoalesceBatchesExec>()
                    {
                        let inner_child = exec_child.children()[0].clone();
                        if let Some(inner_take) = inner_child.as_any().downcast_ref::<TakeExec>() {
                            return Ok(Transformed::yes(Self::collapse_takes(
                                inner_take,
                                outer_take,
                                plan.clone(),
                            )));
                        }
                    }
                }
                Ok(Transformed::no(plan))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "coalesce_take"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rule that eliminates [ProjectionExec] nodes that projects all columns
/// from its input with no additional expressions.
#[derive(Debug)]
pub struct SimplifyProjection;

impl PhysicalOptimizerRule for SimplifyProjection {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                if let Some(proj) = plan.as_any().downcast_ref::<ProjectionExec>() {
                    let children = proj.children();
                    if children.len() != 1 {
                        return Ok(Transformed::no(plan));
                    }

                    let input = children[0];

                    // TODO: we could try to coalesce consecutive projections, something for later
                    // For now, we just keep things simple and only remove NoOp projections

                    // output has different schema, projection needed
                    if input.schema() != proj.schema() {
                        return Ok(Transformed::no(plan));
                    }

                    if proj.expr().iter().enumerate().all(|(index, proj_expr)| {
                        if let Some(expr) = proj_expr.expr.as_any().downcast_ref::<Column>() {
                            // no renaming, no reordering
                            expr.index() == index && expr.name() == proj_expr.alias
                        } else {
                            false
                        }
                    }) {
                        return Ok(Transformed::yes(input.clone()));
                    }
                }
                Ok(Transformed::no(plan))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "simplify_projection"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rule that optimizes `FilterExec -> LanceScanExec` into `FilteredReadExec`.
///
/// This transformation pushes down the filter into the scan, allowing for
/// more efficient query execution by leveraging Lance's native filtering
/// capabilities including scalar indices and predicate pushdown.
#[derive(Debug)]
pub struct PushDownFilterToFilteredRead;

impl PhysicalOptimizerRule for PushDownFilterToFilteredRead {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                // Check if this is a FilterExec node
                let (filter_expr, child) = if let Some(lance_filter) =
                    plan.as_any()
                        .downcast_ref::<crate::io::exec::LanceFilterExec>()
                {
                    // LanceFilterExec has the logical expression stored
                    let expr = lance_filter.expr().clone();
                    let children = lance_filter.children();
                    if children.is_empty() {
                        return Ok(Transformed::no(plan));
                    }
                    (Some(expr), children[0].clone())
                } else {
                    return Ok(Transformed::no(plan));
                };

                // Check if the child is a LanceScanExec
                if let Some(lance_scan) = child
                    .as_any()
                    .downcast_ref::<crate::io::exec::LanceScanExec>()
                {
                    if let Some(filter_expr) = filter_expr {
                        // We have a FilterExec -> LanceScanExec pattern
                        // Convert to FilteredReadExec
                        if let Ok(filtered_read) =
                            Self::create_filtered_read(lance_scan, filter_expr)
                        {
                            return Ok(Transformed::yes(filtered_read));
                        }
                    }
                }

                Ok(Transformed::no(plan))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "push_down_filter_to_filtered_read"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl PushDownFilterToFilteredRead {
    /// Create a FilteredReadExec from a LanceScanExec and a filter expression.
    fn create_filtered_read(
        lance_scan: &crate::io::exec::LanceScanExec,
        filter_expr: Expr,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use crate::io::exec::filtered_read::FilteredReadExec;
        use crate::io::exec::filtered_read::FilteredReadOptions;
        use lance_core::datatypes::Projection;

        let dataset = lance_scan.dataset().clone();
        let fragments = lance_scan.fragments().clone();
        let range = lance_scan.range().clone();
        let projection_schema = lance_scan.projection().as_ref().clone();
        let config = lance_scan.config();

        // Convert LanceScanConfig and projection to FilteredReadOptions
        let mut read_options = FilteredReadOptions::new(
            Projection::empty(dataset.clone()).union_schema(&projection_schema),
        );

        // Set fragments
        read_options = read_options.with_fragments(fragments);

        // Set batch size
        if config.batch_size > 0 {
            read_options = read_options.with_batch_size(config.batch_size as u32);
        }

        // Set fragment readahead
        if let Some(fragment_readahead) = config.fragment_readahead {
            read_options = read_options.with_fragment_readahead(fragment_readahead);
        }

        // Set scan range
        if let Some(scan_range) = range {
            read_options = read_options
                .with_scan_range_before_filter(scan_range)
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }

        // Set deleted rows handling
        if config.with_make_deletions_null {
            read_options = read_options
                .with_deleted_rows()
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }

        // Set the filter
        read_options = read_options.with_filter_plan(
            lance_index::scalar::expression::FilterPlan::new_refine_only(filter_expr),
        );

        // Create the FilteredReadExec
        let filtered_read = FilteredReadExec::try_new(dataset, read_options, None)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        Ok(Arc::new(filtered_read))
    }
}

pub fn get_physical_optimizer() -> PhysicalOptimizer {
    PhysicalOptimizer::with_rules(vec![
        // Push down FilterExec on top of LanceScanExec into FilteredReadExec
        Arc::new(crate::io::exec::optimizer::PushDownFilterToFilteredRead),
        Arc::new(crate::io::exec::optimizer::CoalesceTake),
        Arc::new(crate::io::exec::optimizer::SimplifyProjection),
        // Push down limit into FilteredReadExec and other Execs via with_fetch()
        Arc::new(datafusion::physical_optimizer::limit_pushdown::LimitPushdown::new()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::exec::{LanceScanConfig, LanceScanExec};
    use crate::{io::exec::LanceFilterExec, Dataset};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use datafusion::prelude::*;
    use lance_core::datatypes::Schema;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    /// Test that the PushDownFilterToFilteredRead optimizer rule correctly
    /// transforms a `LanceFilterExec -> LanceScanExec` pattern into `FilteredReadExec`.
    #[tokio::test]
    async fn test_push_down_filter_to_filtered_read() -> Result<(), Box<dyn std::error::Error>> {
        use lance_core::utils::tempfile::TempStrDir;

        // Create a test dataset
        let test_uri = TempStrDir::default();

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_string())));
        Dataset::write(data_gen.batch(100), &test_uri, None).await?;

        let dataset = Dataset::open(&test_uri).await?;
        let fragments = dataset.fragments().clone();
        let lance_schema = dataset.schema().clone();

        // Create a LanceScanExec
        let projection = Arc::new(Schema::try_from(lance_schema.project_by_schema(
            &ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]),
            lance_core::datatypes::OnMissing::Error,
            lance_core::datatypes::OnTypeMismatch::TakeSelf,
        )?)?);

        let scan_config = LanceScanConfig::default();
        let scan_exec = Arc::new(LanceScanExec::new(
            Arc::new(dataset.clone()),
            fragments.clone(),
            None,
            projection.clone(),
            scan_config.clone(),
        ));

        // Create a LanceFilterExec on top of the scan
        let filter_expr = col("i").gt(lit(10));
        let filter_exec = Arc::new(LanceFilterExec::try_new(
            filter_expr.clone(),
            scan_exec.clone(),
        )?);

        // Apply the optimizer rule
        let rule = PushDownFilterToFilteredRead;
        let optimized = rule.optimize(filter_exec, &ConfigOptions::new())?;

        // Verify the result is a FilteredReadExec
        assert_eq!(optimized.name(), "FilteredReadExec");

        // Verify the schema is preserved
        assert_eq!(optimized.schema(), scan_exec.schema());

        Ok(())
    }

    /// Test that the optimizer skips LanceScanExec without a filter.
    #[tokio::test]
    async fn test_optimizer_skips_scan_without_filter() -> Result<(), Box<dyn std::error::Error>> {
        use lance_core::utils::tempfile::TempStrDir;

        // Create a test dataset
        let test_uri = TempStrDir::default();

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_string())));
        Dataset::write(data_gen.batch(100), &test_uri, None).await?;

        let dataset = Dataset::open(&test_uri).await?;
        let fragments = dataset.fragments().clone();
        let lance_schema = dataset.schema().clone();

        // Create a LanceScanExec without a filter
        let projection = Arc::new(Schema::try_from(lance_schema.project_by_schema(
            &ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]),
            lance_core::datatypes::OnMissing::Error,
            lance_core::datatypes::OnTypeMismatch::TakeSelf,
        )?)?);

        let scan_exec = Arc::new(LanceScanExec::new(
            Arc::new(dataset.clone()),
            fragments.clone(),
            None,
            projection.clone(),
            LanceScanConfig::default(),
        ));

        // Apply the optimizer rule
        let rule = PushDownFilterToFilteredRead;
        let optimized = rule.optimize(scan_exec.clone(), &ConfigOptions::new())?;

        // Verify the plan is unchanged (still a LanceScanExec)
        assert_eq!(optimized.name(), "LanceScanExec");

        Ok(())
    }
}
