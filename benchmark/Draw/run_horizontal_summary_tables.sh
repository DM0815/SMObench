#!/bin/bash

# SMOBench Horizontal Styled Summary Table Generator
# Generates scIB-style summary tables for horizontal integration (leiden only)

echo "=== SMOBench Horizontal Styled Summary Table Generator ==="
echo "Starting horizontal styled summary table generation..."

# Ensure output directory exists
if [ ! -d "Results/horizontal_summary_table" ]; then
    echo "Creating Results/horizontal_summary_table directory..."
    mkdir -p Results/horizontal_summary_table
fi

echo "Executing R script for horizontal styled summary table generation..."

Rscript Draw/plotHorizontalSummaryTableStyled.R

if [ $? -eq 0 ]; then
    echo "Success: Horizontal styled summary tables generated!"
    echo "Output directory: Results/horizontal_summary_table/"
    echo ""
    echo "Generated files:"
    ls -la Results/horizontal_summary_table/SMOBench_horizontal_styled_summary_*.png 2>/dev/null || echo "No horizontal summary PNG files found"
    echo ""
    echo "Generated PDF files:"
    ls -la Results/horizontal_summary_table/SMOBench_horizontal_styled_summary_*.pdf 2>/dev/null || echo "No horizontal summary PDF files found"
else
    echo "Error: R script execution failed!"
    echo "Please check the error messages above."
fi

echo "=== Horizontal Styled Summary Generation Complete ==="
