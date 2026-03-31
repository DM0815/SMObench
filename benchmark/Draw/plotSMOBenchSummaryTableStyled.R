#' SMOBench Summary Table - scIB Style Visualization
#' 
#' Adapted from scIB metrics plot for SMOBench vertical integration
#' Creates styled summary tables showing method performance with color-coded sections
#' 
#' Date: 2024

# Load required libraries
library(tidyverse)
library(ggnewscale)

#' Load and process SMOBench detailed results
#' 
#' @param results_dir Directory containing detailed results CSV files
#' @param clustering_method Specific clustering method to process
#' @return Processed data frame ready for summary table
loadSMOBenchResults <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                               clustering_method = "leiden") {
  
  file_path <- file.path(results_dir, paste0("detailed_results_", clustering_method, ".csv"))
  
  if (!file.exists(file_path)) {
    stop("File not found: ", file_path)
  }
  
  # Read CSV and preserve column names with spaces
  data <- read.csv(file_path, stringsAsFactors = FALSE, check.names = FALSE)
  
  cat("Loaded detailed results for", clustering_method, ":\n")
  cat("  Rows:", nrow(data), "\n")
  cat("  Methods:", length(unique(data$Method)), "\n") 
  cat("  Datasets:", length(unique(data$Dataset)), "\n")
  
  return(data)
}

#' Create aggregated data for specific combination (RNA+ADT/ATAC + withGT/woGT)
#' 
#' @param detailed_data Raw detailed results data
#' @param datatype_filter "RNA+ADT" or "RNA+ATAC"
#' @param gt_filter "True" or "False" (string)
#' @return Aggregated data with mean scores across datasets
createAggregatedData <- function(detailed_data, datatype_filter, gt_filter) {
  
  # Create datatype mapping
  detailed_data <- detailed_data %>%
    mutate(
      Datatype = case_when(
        str_detect(Dataset_Type, "RNA_ADT") ~ "RNA+ADT",
        str_detect(Dataset_Type, "RNA_ATAC") ~ "RNA+ATAC", 
        TRUE ~ Dataset_Type
      )
    )
  
  # Filter for specific combination
  filtered_data <- detailed_data %>%
    filter(Datatype == datatype_filter, GT_Available == gt_filter)
  
  if (nrow(filtered_data) == 0) {
    cat("Warning: No data found for", datatype_filter, "+", gt_filter, "\n")
    return(NULL)
  }
  
  cat("Processing", datatype_filter, "+", gt_filter, ":\n")
  cat("  Found", nrow(filtered_data), "rows across", length(unique(filtered_data$Dataset)), "datasets\n")
  cat("  Datasets:", paste(unique(filtered_data$Dataset), collapse = ", "), "\n")
  
  # Determine metrics based on GT status
  if (gt_filter == "True") {
    # withGT metrics: SC + BioC (ARI, NMI, ASW celltype, Graph cLISI)
    aggregated <- filtered_data %>%
      group_by(Method) %>%
      summarise(
        # Spatial Coherence (1 metric) - Use Moran's Index instead of SC_Score
        `Moran's Index` = mean(`Moran Index`, na.rm = TRUE),
        # Biological Conservation (4 metrics)
        ARI = mean(ARI, na.rm = TRUE),
        NMI = mean(NMI, na.rm = TRUE),
        `ASW celltype` = mean(asw_celltype, na.rm = TRUE),
        `Graph cLISI` = mean(graph_clisi, na.rm = TRUE),
        # Aggregate scores
        `Spatial Coherence` = mean(SC_Score, na.rm = TRUE),
        `Bio Conservation` = mean(BioC_Score, na.rm = TRUE),
        Total = mean(Total_Score, na.rm = TRUE),
        .groups = "drop"
      )
  } else {
    # woGT metrics: SC + woGT BioC metrics (unsupervised quality metrics)
    aggregated <- filtered_data %>%
      group_by(Method) %>%
      summarise(
        # Spatial Coherence (1 metric) - Use Moran's Index instead of SC_Score
        `Moran's Index` = mean(`Moran Index`, na.rm = TRUE),
        # woGT BioC metrics (unsupervised quality metrics, use normalized versions)
        `Silhouette Coefficient` = mean(`Silhouette Coefficient`, na.rm = TRUE),
        `Calinski-Harabasz Index` = mean(`Calinski-Harabaz Index_normalized`, na.rm = TRUE),
        `Davies-Bouldin Index` = mean(`Davies-Bouldin Index_normalized`, na.rm = TRUE),
        # Aggregate scores
        `Spatial Coherence` = mean(SC_Score, na.rm = TRUE),
        `Bio Conservation` = mean(BioC_Score, na.rm = TRUE),
        Total = mean(Total_Score, na.rm = TRUE),
        .groups = "drop"
      )
  }
  
  # Sort by Total score (descending)
  aggregated <- aggregated %>%
    arrange(desc(Total))
  
  return(aggregated)
}

#' Create scIB-style summary table plot
#' 
#' @param aggregated_data Aggregated data from createAggregatedData
#' @param title_suffix Title suffix for the plot
#' @return ggplot object
createStyledSummaryTable <- function(aggregated_data, title_suffix = "") {
  
  if (is.null(aggregated_data) || nrow(aggregated_data) == 0) {
    return(NULL)
  }
  
  # Prepare data for plotting
  col_order <- names(aggregated_data)
  plot_data <- aggregated_data %>%
    mutate(Method = factor(Method, levels = rev(Method))) %>%
    pivot_longer(-Method, names_to="Metric", values_to="Score") %>%
    mutate(
      x = as.numeric(factor(Metric, levels = col_order)),
      y = as.numeric(Method),
      section = case_when(
        Metric == "Moran's Index" ~ "Spatial Coherence",
        Metric %in% c("ARI", "NMI", "ASW celltype", "Graph cLISI", 
                     "Silhouette Coefficient", "Calinski-Harabasz Index", "Davies-Bouldin Index") ~ "Biological Conservation",
        Metric %in% c("Spatial Coherence", "Bio Conservation", "Total") ~ "Aggregate Score",
        TRUE ~ "Other"
      ),
      shape_type = ifelse(section == "Aggregate Score", "bar", "circle")
    ) %>%
    group_by(Metric) %>%
    mutate(
      metric_max = suppressWarnings(max(Score, na.rm = TRUE)),
      metric_min = suppressWarnings(min(Score, na.rm = TRUE)),
      range_val = metric_max - metric_min,
      range_val = ifelse(is.finite(range_val) & range_val > 0, range_val, 1),
      Score_scaled = ifelse(is.finite(Score), (Score - metric_min) / range_val, 0)
    ) %>%
    ungroup() %>%
    mutate(
      color_value = ifelse(section == "Aggregate Score", Score, Score_scaled)
    )
  
  # Count sections for header positioning (adjust based on GT status)
  n_sc = 1  # Moran's Index
  # Dynamic BioC count: withGT=4 (ARI, NMI, ASW celltype, Graph cLISI), woGT=3 (Silhouette, Calinski, Davies-Bouldin)  
  n_bioc = length(unique(plot_data$Metric[plot_data$section == "Biological Conservation"]))
  n_agg = 3   # Spatial Coherence, Bio Conservation, Total
  
  # Header data - Top level (2 rows)
  header_data_top <- tibble(
    # x = c(1, 1 + n_sc + n_bioc/2, 1 + n_sc + n_bioc + n_agg/2) + 1.2,
    x = c(2.1, 2 + n_sc + n_bioc/2-0.25, 2 + n_sc + n_bioc + n_agg/2-0.25),
    y = nrow(aggregated_data) + 2.7,
    label = c("Spatial\nCoherence", "Biological\nConservation", "Aggregate\nScore")
  )

  
  header_data_method <- tibble(x = 0.95, y = nrow(aggregated_data) + 1.15, label = "Method")
  
  header_data_sub <- tibble(
    x = 2:length(col_order),
    y = nrow(aggregated_data) + 1.15,
    label = str_wrap(col_order[-1], width = 8)
  )
  
  # Line data
  line_data_full <- tibble(
    x = 0.5, xend = length(col_order) + 0.5, 
    y = nrow(aggregated_data) + 0.5, yend = nrow(aggregated_data) + 0.5
  )
  
  line_data_part <- tibble(
    x = c(1.5, 1.5 + n_sc, 1.5 + n_sc + n_bioc), 
    xend = c(1.5 + n_sc, 1.5 + n_sc + n_bioc, 1.5 + n_sc + n_bioc + n_agg), 
    y = nrow(aggregated_data) + 2.2, yend = nrow(aggregated_data) + 2.2
  )
  
  # Create plot
  p <- ggplot() +
    # ① Spatial Coherence (orange) - improved contrast
    geom_point(data = plot_data %>% filter(section == "Spatial Coherence"),
               aes(x = x, y = y, fill = color_value), shape = 21, size = 12, color = "white", stroke = 0.5) +
    scale_fill_gradient(low = "#fdd49e", high = "#d94801", limits = c(0, 1)) +
    new_scale_fill() +
    
    # ② Biological Conservation (green) - improved contrast
    geom_point(data = plot_data %>% filter(section == "Biological Conservation"),
               aes(x = x, y = y, fill = color_value), shape = 21, size = 12, color = "white", stroke = 0.5) +
    scale_fill_gradient(low = "#c7e9c0", high = "#238b45", limits = c(0, 1)) +
    new_scale_fill() +
    
    # ③ Aggregate Score (blue) — bar proportional to value, improved contrast
    geom_tile(data = plot_data %>% filter(section == "Aggregate Score"),
              aes(x = x - (1 - Score)/2, y = y, width = Score, height = 0.55, fill = color_value),
              color = "white", linewidth = 0.5) +
    scale_fill_gradient(low = "#c6dbef", high = "#08519c", limits = c(0, 1)) +
    
    # Labels for all
    # Replace your geom_text() with these two
    geom_text(
      data = plot_data %>% filter(section != "Aggregate Score"),
      aes(x = x, y = y, label = sprintf("%.2f", Score)),
      size = 3.3, hjust = 0.5  # centered (default)
    ) +
    geom_text(
      data = plot_data %>% filter(section == "Aggregate Score"),
      aes(x = x - 0.4, y = y, label = sprintf("%.2f", Score)),
      size = 3.3, hjust = 0    # left-aligned
    ) +  
    # Headers
    geom_text(data = header_data_top, aes(x = x-0.3, y = y, label = label), size = 5, fontface = "bold") +
    geom_text(data = header_data_method, aes(x = x, y = y, label = label), size = 3.8) +
    geom_text(data = header_data_sub, aes(x = x, y = y, label = label), size = 3.6, lineheight = 0.9) +
    
    # Lines
    geom_segment(data = line_data_full, aes(x = x, xend = xend, y = y, yend = yend)) +
    geom_segment(data = line_data_part, aes(x = x, xend = xend, y = y, yend = yend)) +
    
    # Axis setup
    scale_y_continuous(breaks = 1:nrow(aggregated_data), labels = levels(plot_data$Method), expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    coord_cartesian(xlim = c(0.5, length(col_order) + 0.5), ylim = c(0.5, nrow(aggregated_data) + 3.5), clip = "off") +
    
    # Theme
    theme_void() +
    theme(
      axis.text.y = element_text(hjust = 0.5, face = "bold", margin = margin(r = -70)),
      legend.position = "none",
      plot.margin = margin(20, 20, 20, 20),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    ) +
    labs(title = paste("SMOBench Performance Summary", title_suffix))
  
  return(p)
}

#' Generate all styled summary tables
#' 
#' @param results_dir Directory containing detailed results
#' @param output_dir Output directory for plots
#' @param clustering_methods Vector of clustering methods to process
generateStyledSummaryTables <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                                       output_dir = "Results/summary_table/",
                                       clustering_methods = c("leiden", "louvain", "kmeans", "mclust")) {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Define combinations
  combinations <- list(
    list(datatype = "RNA+ADT", gt_status = "True", name = "RNA_ADT_withGT", title = "RNA+ADT withGT"),
    list(datatype = "RNA+ADT", gt_status = "False", name = "RNA_ADT_woGT", title = "RNA+ADT woGT"),
    list(datatype = "RNA+ATAC", gt_status = "True", name = "RNA_ATAC_withGT", title = "RNA+ATAC withGT"),
    list(datatype = "RNA+ATAC", gt_status = "False", name = "RNA_ATAC_woGT", title = "RNA+ATAC woGT")
  )
  
  # Process each clustering method
  for (clustering in clustering_methods) {
    
    cat("\nProcessing clustering method:", clustering, "\n")
    
    # Load data
    detailed_data <- loadSMOBenchResults(results_dir, clustering)
    
    # Create tables for each combination
    for (combo in combinations) {
      
      cat("\nCreating", combo$name, "table...\n")
      
      # Create aggregated data
      aggregated_data <- createAggregatedData(detailed_data, combo$datatype, combo$gt_status)
      
      if (!is.null(aggregated_data)) {
        # Create styled plot
        plot_title <- paste(combo$title, "-", stringr::str_to_title(clustering))
        styled_plot <- createStyledSummaryTable(aggregated_data, plot_title)
        
        if (!is.null(styled_plot)) {
          # Save plot
          filename_base <- file.path(output_dir, paste0("SMOBench_styled_summary_", combo$name, "_", clustering))
          
          # Save in multiple formats
          ggsave(paste0(filename_base, ".png"), plot = styled_plot, width = 12, height = 7, dpi = 300, bg = "white")
          ggsave(paste0(filename_base, ".pdf"), plot = styled_plot, width = 12, height = 7, bg = "white")
          
          cat("Saved styled summary table:", filename_base, "\n")
        }
      }
    }
  }
  
  cat("\nStyled summary table generation complete!\n")
  cat("Output directory:", output_dir, "\n")
}

# Main execution
if (!interactive()) {
  generateStyledSummaryTables()
} else {
  cat("SMOBench Styled Summary Table functions loaded.\n")
  cat("Run generateStyledSummaryTables() to create all styled tables.\n")
}
