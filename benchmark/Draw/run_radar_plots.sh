#!/bin/bash

# SMOBench Radar Plot Generator
# Generates radar charts for all datasets showing method performance across clustering algorithms

echo "=== SMOBench Radar Chart Generator ==="
echo "Starting radar chart generation..."

# Check if Results/plots directory exists, create if not
if [ ! -d "Results/plots" ]; then
    echo "Creating Results/plots directory..."
    mkdir -p Results/plots
fi

echo "Executing R script for radar chart generation..."

# Run the R script
Rscript Draw/plotSMOBenchRadar.R

if [ $? -eq 0 ]; then
    echo "Success: Radar charts generated!"
    echo "Output directory: Results/plots/"
    echo ""
    echo "Generated files:"
    ls -la Results/plots/SMOBench_radar_*.png 2>/dev/null || echo "No radar chart files found"
else
    echo "Error: R script execution failed!"
    echo "Please check the error messages above"
fi

echo "=== Radar Chart Generation Complete ==="