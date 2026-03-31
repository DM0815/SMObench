#!/bin/bash

# SMOBench Scatter Plot Generation Script
# 
# This script generates summary scatter plots for SMOBench vertical integration results
# Run this from the SMOBench-CLEAN root directory

echo "=== SMOBench Scatter Plot Generator ==="
echo "Starting scatter plot generation..."
echo

# Check if we're in the right directory
if [ ! -d "Results/evaluation/vertical_integration/final_results" ]; then
    echo "Error: Results directory not found!"
    echo "Please run this script from the SMOBench-CLEAN root directory"
    echo "Expected directory: Results/evaluation/vertical_integration/final_results/"
    exit 1
fi

# Check if R is available
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript not found!"
    echo "Please install R and ensure Rscript is in your PATH"
    exit 1
fi

# Create plots directory if it doesn't exist
mkdir -p Results/plots

# Run the R script
echo "Executing R script for scatter plot generation..."
Rscript Draw/plotSMOBenchScatter.R

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo
    echo "=== Scatter plot generation completed successfully! ==="
    echo
    echo "Generated plots:"
    echo "- Summary Scatter plots: Individual clustering method plots (leiden, louvain, kmeans, mclust)"
    echo "- Best Methods Scatter plots: Aggregated performance plots with error bars"
    echo "- Overall clustering comparison plot"
    echo
    echo "Output location: Results/plots/"
    echo "File formats: PDF, PNG, TIFF"
    echo
    ls -la Results/plots/*.pdf 2>/dev/null || echo "PDF files will be available after execution"
else
    echo
    echo "Error: R script execution failed!"
    echo "Please check the error messages above"
    exit 1
fi