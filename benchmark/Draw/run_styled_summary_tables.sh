#!/bin/bash

# SMOBench Styled Summary Table Generator
# Creates scIB-style summary tables for SMOBench vertical integration results

echo "=== SMOBench Styled Summary Table Generator ==="
echo "Starting styled summary table generation..."

# Check if Results/summary_table directory exists, create if not
if [ ! -d "Results/summary_table" ]; then
    echo "Creating Results/summary_table directory..."
    mkdir -p Results/summary_table
fi

echo "Executing R script for styled summary table generation..."

# Run the R script  
Rscript Draw/plotSMOBenchSummaryTableStyled.R

if [ $? -eq 0 ]; then
    echo "Success: Styled summary tables generated!"
    echo "Output directory: Results/summary_table/"
    echo ""
    echo "Generated files:"
    ls -la Results/summary_table/SMOBench_styled_summary_*.png 2>/dev/null || echo "No styled summary table files found"
    echo ""
    echo "Generated PDF files:"
    ls -la Results/summary_table/SMOBench_styled_summary_*.pdf 2>/dev/null || echo "No styled summary PDF files found"
else
    echo "Error: R script execution failed!"
    echo "Please check the error messages above"
fi

echo "=== Styled Summary Table Generation Complete ==="