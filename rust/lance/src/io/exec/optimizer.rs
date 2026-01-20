// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Physical Optimizer Rules

use std::sync::Arc;

use super::{
    filter::LanceFilterExec,
    filtered_read::{FilteredReadExec, FilteredReadOptions},
    LanceScanExec, TakeExec,
};
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DFResult,
    logical_expr::Expr,
    physical_optimizer::{optimizer::PhysicalOptimizer, PhysicalOptimizerRule},
    physical_plan::{
        coalesce_batches::CoalesceBatchesExec, projection::ProjectionExec, ExecutionPlan,
    },
};
use datafusion_physical_expr::{expressions::Column, PhysicalExpr};
use lance_core::datatypes::{OnMissing, Projection};

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

/// Rule that converts `LanceScanExec -> FilterExec` or `LanceScanExec -> LanceFilterExec` into `FilteredReadExec`.
///
/// This optimization pushes the filter down into the scan, which allows Lance
/// to use more efficient scanning strategies such as skipping fragments or using
/// indices.
#[derive(Debug)]
pub struct ConvertScanFilterToFilteredRead;

impl ConvertScanFilterToFilteredRead {
    /// Try to extract a logical expression from a filter execution plan
    fn try_extract_filter_expr(plan: &Arc<dyn ExecutionPlan>) -> Option<Expr> {
        // Try LanceFilterExec which stores the logical expression directly
        if let Some(lance_filter) = plan.as_any().downcast_ref::<LanceFilterExec>() {
            return Some(lance_filter.expr().clone());
        }

        None
    }
}

impl PhysicalOptimizerRule for ConvertScanFilterToFilteredRead {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                // Try to extract filter expression from the plan
                let Some(expr) = Self::try_extract_filter_expr(&plan) else { return Ok(Transformed::no(plan)) };

                // Get the child of the filter
                let children = plan.children();
                if children.len() != 1 {
                    return Ok(Transformed::no(plan));
                }
                let child = children[0].clone();

                // Check if the child is a LanceScanExec
                let Some(scan_exec) = child.as_any().downcast_ref::<LanceScanExec>() else { return Ok(Transformed::no(plan)) };

                // Create FilteredReadOptions from the LanceScanExec output schema.
                // We use union_arrow_schema on the scan's output schema (ArrowSchema) which
                // includes both the projected columns and any system columns (_rowid, _rowaddr, etc.).
                // This automatically detects and sets the system column flags based on field names.
                let projection = match Projection::empty(scan_exec.dataset().clone())
                    .union_arrow_schema(scan_exec.schema().as_ref(), OnMissing::Error)
                {
                    Ok(proj) => proj,
                    Err(e) => {
                        log::debug!(
                            "Failed to create projection from scan schema: {:?}. Falling back to original plan.",
                            e
                        );
                        return Ok(Transformed::no(plan));
                    }
                };
                let mut filtered_read_options = FilteredReadOptions::new(projection);

                // Set the filter
                filtered_read_options.full_filter = Some(expr);

                // Copy fragments from the scan
                filtered_read_options.fragments = Some(scan_exec.fragments().clone());

                // Copy range from the scan if it was specified
                if let Some(range) = scan_exec.range().clone() {
                    filtered_read_options.scan_range_before_filter = Some(range);
                }

                // Copy batch size if specified (LanceScanConfig has batch_size as usize, FilteredReadOptions expects u32)
                let batch_size = scan_exec.config().batch_size;
                filtered_read_options.batch_size = Some(batch_size as u32);

                // Create the FilteredReadExec
                match FilteredReadExec::try_new(
                    scan_exec.dataset().clone(),
                    filtered_read_options,
                    None, // No index input
                ) {
                    Ok(filtered_read_exec) => {
                        Ok(Transformed::yes(Arc::new(filtered_read_exec)))
                    }
                    Err(_) => {
                        // If we can't create FilteredReadExec, fall back to the original plan
                        Ok(Transformed::no(plan))
                    }
                }
            })?
            .data)
    }

    fn name(&self) -> &str {
        "convert_scan_filter_to_filtered_read"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

pub fn get_physical_optimizer() -> PhysicalOptimizer {
    PhysicalOptimizer::with_rules(vec![
        Arc::new(crate::io::exec::optimizer::CoalesceTake),
        Arc::new(crate::io::exec::optimizer::SimplifyProjection),
        Arc::new(crate::io::exec::optimizer::ConvertScanFilterToFilteredRead),
        // Push down limit into FilteredReadExec and other Execs via with_fetch()
        Arc::new(datafusion::physical_optimizer::limit_pushdown::LimitPushdown::new()),
    ])
}

#[cfg(test)]
mod tests {
    use super::super::LanceScanConfig;
    use super::*;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use crate::Dataset;
    use datafusion::common::config::ConfigOptions;
    use datafusion::physical_plan::filter::FilterExec;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::Result;
    use lance_datafusion::planner::Planner;
    use lance_datagen::{array, ByteCount};
    use std::sync::Arc;

    /// Helper to create a dataset in async context (without creating a new runtime)
    async fn create_test_dataset() -> Result<(TempStrDir, Arc<Dataset>)> {
        let tmp_dir = TempStrDir::default();
        let dataset = lance_datagen::gen_batch()
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_dataset(
                tmp_dir.as_str(),
                FragmentCount::from(4),
                FragmentRowCount::from(100),
            )
            .await?;
        Ok((tmp_dir, Arc::new(dataset)))
    }

    /// Test that LanceScanExec -> LanceFilterExec is converted to FilteredReadExec
    #[test_log::test(tokio::test)]
    async fn test_convert_scan_lance_filter_to_filtered_read() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec with a simple filter expression
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // Check that the result is a FilteredReadExec
        assert!(optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test that LanceScanExec -> LanceFilterExec with range is converted properly
    #[test_log::test(tokio::test)]
    async fn test_convert_with_scan_range() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec with a range
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            Some(10..20),
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // Check that the result is a FilteredReadExec
        let filtered_read = optimized
            .as_any()
            .downcast_ref::<FilteredReadExec>()
            .unwrap();
        assert_eq!(
            filtered_read.options().scan_range_before_filter,
            Some(10..20)
        );
        Ok(())
    }

    /// Test that LanceScanExec -> LanceFilterExec with batch_size is converted properly
    #[test_log::test(tokio::test)]
    async fn test_convert_with_batch_size() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        let config = LanceScanConfig {
            batch_size: 512,
            ..Default::default()
        };

        // Create a LanceScanExec with a custom batch_size
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            config,
        ));

        // Create a LanceFilterExec
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // Check that the result is a FilteredReadExec with correct batch_size
        let filtered_read = optimized
            .as_any()
            .downcast_ref::<FilteredReadExec>()
            .unwrap();
        assert_eq!(filtered_read.options().batch_size, Some(512));
        Ok(())
    }

    /// Test that regular FilterExec is NOT converted (because we can't reliably
    /// convert physical expressions back to logical expressions)
    #[test_log::test(tokio::test)]
    async fn test_no_convert_for_regular_filter_exec() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Create a regular FilterExec (not LanceFilterExec)
        let planner = Planner::new(scan.schema());
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let predicate = planner.create_physical_expr(&expr)?;
        let filter_exec = FilterExec::try_new(predicate, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter_exec);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // The result should still be a FilterExec, not converted to FilteredReadExec
        assert!(optimized.as_any().is::<FilterExec>());
        assert!(!optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test that plan with children other than LanceScanExec is not converted
    #[test_log::test(tokio::test)]
    async fn test_no_convert_for_non_scan_child() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Wrap the scan in a projection
        let expr = datafusion::physical_expr::expressions::col("text", &scan.schema())?;
        let proj = ProjectionExec::try_new(vec![(expr, "text".to_string())], scan)?;
        let proj_plan: Arc<dyn ExecutionPlan> = Arc::new(proj);

        // Create a LanceFilterExec on top of the projection (not the scan)
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let filter = LanceFilterExec::try_new(expr, proj_plan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // The result should still be a LanceFilterExec, not converted to FilteredReadExec
        assert!(optimized.as_any().is::<LanceFilterExec>());
        assert!(!optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test that LanceScanExec without filter is not modified
    #[test_log::test(tokio::test)]
    async fn test_no_convert_without_filter() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec without any filter
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Apply the optimizer rule directly to the scan (no filter)
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(scan, &ConfigOptions::new())?;

        // The result should still be a LanceScanExec, not converted
        assert!(optimized.as_any().is::<LanceScanExec>());
        assert!(!optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test conversion with a complex filter expression
    #[test_log::test(tokio::test)]
    async fn test_convert_with_complex_filter() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec with a complex filter expression
        let expr = datafusion::logical_expr::col("text")
            .is_not_null()
            .and(datafusion::logical_expr::lit(true));
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // Check that the result is a FilteredReadExec
        assert!(optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test conversion preserves fragments from the original scan
    #[test_log::test(tokio::test)]
    async fn test_convert_preserves_fragments() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;
        let original_fragments = dataset.fragments().clone();

        // Create a LanceScanExec
        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            original_fragments.clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec
        let expr = datafusion::logical_expr::col("text").is_not_null();
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // Check that the result is a FilteredReadExec with the same fragments
        let filtered_read = optimized
            .as_any()
            .downcast_ref::<FilteredReadExec>()
            .unwrap();
        assert_eq!(
            filtered_read.options().fragments.as_ref().unwrap().len(),
            original_fragments.len()
        );
        Ok(())
    }

    /// Test that the optimizer falls back to original plan when projection creation fails
    #[test_log::test(tokio::test)]
    async fn test_fallback_when_projection_creation_fails() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec with a schema that includes a non-existent field
        // This will cause Projection::union_arrow_schema to fail because the field
        // doesn't exist in the dataset's schema
        use lance_core::datatypes::Schema as LanceSchema;
        let non_existent_field = lance_core::datatypes::Field::new_arrow(
            "non_existent_column",
            arrow_schema::DataType::Utf8,
            true,
        )?;
        let mut schema = LanceSchema::default();
        schema.fields.push(non_existent_field);

        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(schema),
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec using the field that exists in the schema
        let expr = datafusion::logical_expr::col("non_existent_column").is_not_null();
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // The result should still be a LanceFilterExec (fallback to original plan)
        assert!(optimized.as_any().is::<LanceFilterExec>());
        assert!(!optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }

    /// Test that the optimizer falls back to original plan when FilteredReadExec creation fails
    #[test_log::test(tokio::test)]
    async fn test_fallback_when_filtered_read_creation_fails() -> Result<()> {
        let (_tmp_dir, dataset) = create_test_dataset().await?;

        // Create a LanceScanExec with an empty schema
        // This will cause FilteredReadExec::try_new to fail because projection is empty
        // and there are no system columns
        use lance_core::datatypes::Schema as LanceSchema;
        let empty_schema = Arc::new(LanceSchema::default());

        let scan = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            empty_schema,
            LanceScanConfig::default(),
        ));

        // Create a LanceFilterExec using a literal expression (doesn't reference any field)
        let expr = datafusion::logical_expr::lit(true);
        let filter = LanceFilterExec::try_new(expr, scan)?;
        let filter_plan: Arc<dyn ExecutionPlan> = Arc::new(filter);

        // Apply the optimizer rule
        let rule = ConvertScanFilterToFilteredRead;
        let optimized = rule.optimize(filter_plan, &ConfigOptions::new())?;

        // The result should still be a LanceFilterExec (fallback to original plan)
        assert!(optimized.as_any().is::<LanceFilterExec>());
        assert!(!optimized.as_any().is::<FilteredReadExec>());
        Ok(())
    }
}
