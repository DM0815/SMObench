#' SMOBench Radar Chart Visualization
#' 
#' Creates radar charts for each dataset showing method performance across 
#' different clustering algorithms
#' 
#' Date: 2024

# Load required libraries
library(fmsb)
library(scales)

#' Load summary data for all clustering methods
#' 
#' @param results_dir Directory containing summary CSV files
#' @return List of data frames for each clustering method
loadSummaryData <- function(results_dir = "Results/evaluation/vertical_integration/final_results/") {
  
  clustering_methods <- c("leiden", "louvain", "kmeans", "mclust")
  summary_data <- list()
  
  for (clustering in clustering_methods) {
    file_path <- file.path(results_dir, paste0("summary_", clustering, ".csv"))
    
    if (file.exists(file_path)) {
      data <- read.csv(file_path, row.names = 1, check.names = FALSE)
      # Remove the Average column as we want individual dataset scores
      data <- data[, !colnames(data) %in% "Average"]
      summary_data[[clustering]] <- data
      cat("Loaded data for", clustering, "clustering:", nrow(data), "methods x", ncol(data), "datasets\n")
    } else {
      cat("Warning: File not found:", file_path, "\n")
    }
  }
  
  return(summary_data)
}

#' Prepare radar chart data for a specific dataset
#' 
#' @param summary_data List of clustering method data
#' @param dataset_name Name of the dataset (column name)
#' @return Data frame formatted for fmsb radarchart
prepareRadarData <- function(summary_data, dataset_name) {
  
  # Derive the union of methods across all clustering summaries to ensure consistency
  # Determine all methods present across clustering summaries
  method_sets <- lapply(summary_data, rownames)
  methods <- unique(unlist(method_sets))
  # Preserve the ordering from the first available summary, append any remaining in alphabetical order
  if (length(method_sets) > 0) {
    primary_order <- method_sets[[1]]
    extra_methods <- setdiff(methods, primary_order)
    methods <- c(primary_order, sort(extra_methods))
  } else {
    methods <- character(0)
  }
  clustering_methods <- names(summary_data)
  
  # Create data frame with methods as columns
  radar_data <- data.frame(row.names = clustering_methods)
  
  # Extract scores for each method and clustering algorithm
  for (method in methods) {
    method_scores <- numeric(length(clustering_methods))
    names(method_scores) <- clustering_methods
    
    for (i in seq_along(clustering_methods)) {
      clustering <- clustering_methods[i]
      if (method %in% rownames(summary_data[[clustering]]) && 
          dataset_name %in% colnames(summary_data[[clustering]])) {
        score <- summary_data[[clustering]][method, dataset_name]
        # Handle missing values
        method_scores[i] <- ifelse(is.na(score) || score == "", 0, as.numeric(score))
      } else {
        method_scores[i] <- 0
      }
    }
    
    radar_data[[method]] <- method_scores
  }
  
  # Add max and min rows (required by fmsb)
  # Determine reasonable max based on data range
  max_val <- 1.0  # Assuming scores are normalized to [0,1]
  min_val <- 0.0
  
  radar_data <- rbind(
    rep(max_val, ncol(radar_data)),  # Max row
    rep(min_val, ncol(radar_data)),  # Min row  
    radar_data                       # Actual data
  )
  
  return(radar_data)
}

#' Create radar chart for a single dataset
#' 
#' @param radar_data Prepared data from prepareRadarData
#' @param dataset_name Name of the dataset
#' @param output_path Path to save the plot
createRadarChart <- function(radar_data, dataset_name, output_path) {
  
  # Define colors for the four clustering methods
  clustering_colors <- c(
    "leiden" = "#E31A1C",      # Red
    "louvain" = "#1F78B4",     # Blue
    "kmeans" = "#33A02C",      # Green  
    "mclust" = "#FF7F00"       # Orange
  )
  
  # Create fill colors with transparency
  fill_colors <- alpha(clustering_colors, 0.25)
  line_colors <- clustering_colors
  
  # Create PNG file
  png(output_path, width = 800, height = 800, res = 120)
  
  # Set margins for title
  par(mar = c(1, 1, 4, 1))
  
  # Create radar chart
  radarchart(
    radar_data,
    axistype = 1,
    
    # Polygon options  
    pcol = line_colors,
    pfcol = fill_colors,
    plwd = 2,
    plty = 1,
    
    # Grid options
    cglcol = "grey70",
    cglty = 1,
    axislabcol = "grey40", 
    caxislabels = seq(0, 1, by = 0.2),
    cglwd = 0.8,
    
    # Label options
    vlcex = 1.1,
    title = paste("SMOBench Performance -", dataset_name)
  )
  
  # Add legend
  legend(
    x = "topright",
    legend = names(clustering_colors),
    col = clustering_colors,
    lty = 1,
    lwd = 2,
    cex = 0.9,
    bty = "n"
  )
  
  # Close and save file
  dev.off()
  
  cat("Saved radar chart for", dataset_name, "to", output_path, "\n")
}

#' Generate radar charts for all datasets
#' 
#' @param results_dir Directory containing summary CSV files
#' @param output_dir Directory to save radar charts
generateAllRadarCharts <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                                   output_dir = "Results/plots/") {
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Load all summary data
  cat("Loading summary data for all clustering methods...\n")
  summary_data <- loadSummaryData(results_dir)
  
  if (length(summary_data) == 0) {
    cat("Error: No summary data found!\n")
    return()
  }
  
  # Get all dataset names (should be consistent across clustering methods)
  dataset_names <- colnames(summary_data[[1]])
  cat("Found datasets:", paste(dataset_names, collapse = ", "), "\n")
  
  # Generate radar chart for each dataset
  for (dataset in dataset_names) {
    cat("\nProcessing dataset:", dataset, "\n")
    
    # Prepare data for this dataset
    radar_data <- prepareRadarData(summary_data, dataset)
    
    # Create output filename
    safe_name <- gsub("[^A-Za-z0-9_]", "_", dataset)  # Replace special chars
    output_path <- file.path(output_dir, paste0("SMOBench_radar_", safe_name, ".png"))
    
    # Create radar chart
    createRadarChart(radar_data, dataset, output_path)
  }
  
  cat("\nRadar chart generation complete!\n")
  cat("Charts saved to:", output_dir, "\n")
}

# Main execution
if (!interactive()) {
  generateAllRadarCharts()
} else {
  cat("SMOBench Radar Chart functions loaded.\n")
  cat("Run generateAllRadarCharts() to create all charts.\n")
}
