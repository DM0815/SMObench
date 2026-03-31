#!/usr/bin/env Rscript

# SMOBench Summary Scatter Plot
# Adapted from scib-reproducibility visualization framework
# 
# This script creates summary scatter plots for SMOBench vertical integration results
# showing Spatial Coherence (SC) vs Biological Conservation (BioC) trade-offs
# across different clustering methods and datasets

library(ggplot2)
library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(fs)
library(glue)
library(magrittr)

#' SMOBench Summary Scatter Plot Generator
#' 
#' Creates scatter plots showing SC vs BioC performance for all clustering methods
#' 
#' @param results_dir Path to Results/evaluation/vertical_integration/final_results/
#' @param out_dir Output directory for plots
#' @param clustering_methods Vector of clustering methods to process
#' @param weight_sc Weight for spatial coherence in overall score (default: 0.5)
#' 
makeSMOBenchScatter <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                                out_dir = "Results/plots/",
                                clustering_methods = c("leiden", "louvain", "kmeans", "mclust"),
                                weight_sc = 0.5) {
  
  # Create output directory if it doesn't exist
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  # Process each clustering method
  for (clustering in clustering_methods) {
    
    cat("Processing clustering method:", clustering, "\n")
    
    # Load and process data
    scores <- loadSMOBenchScores(results_dir, clustering)
    
    if (nrow(scores) == 0) {
      cat("Warning: No data found for clustering method:", clustering, "\n")
      next
    }
    
    # Create the plot
    scatter_plot <- plotSMOBenchScatter(scores, getSMOBenchMethodsPal())
    
    # Save plots in multiple formats
    for (format in c("pdf", "png", "tiff")) {
      filename <- glue("{out_dir}/SMOBench_scatter_{clustering}.{format}")
      
      ggsave(
        filename,
        scatter_plot,
        width = 297, height = 210, units = "mm",  # A4 landscape for better facet display
        dpi = 300
      )
    }
    
    cat("Saved plots for", clustering, "clustering\n")
  }
  
  cat("SMOBench scatter plots generation complete!\n")
  cat("Output directory:", out_dir, "\n")
}

#' Load SMOBench scores from detailed results
#' 
#' @param results_dir Path to results directory
#' @param clustering Clustering method name
#' 
#' @return A tibble with processed scores
loadSMOBenchScores <- function(results_dir, clustering) {
  
  # Construct file path
  detailed_file <- file.path(results_dir, glue("detailed_results_{clustering}.csv"))
  
  if (!file.exists(detailed_file)) {
    cat("Warning: File not found:", detailed_file, "\n")
    return(tibble())
  }
  
  # Read the detailed results
  scores_raw <- read_csv(
    detailed_file,
    col_types = cols(
      Method = col_character(),
      Dataset = col_character(),
      Dataset_Type = col_character(),
      Clustering = col_character(),
      GT_Available = col_logical(),
      .default = col_double()
    ),
    show_col_types = FALSE
  )
  
  cat("=== LOAD DATA DEBUG START ===\n")
  cat("Raw data loaded from:", detailed_file, "\n")
  cat("  Raw rows:", nrow(scores_raw), "\n")
  cat("  Raw columns:", ncol(scores_raw), "\n")
  
  # Check raw Dataset_Type values
  cat("\nRaw Dataset_Type values:\n")
  dataset_type_counts <- table(scores_raw$Dataset_Type)
  print(dataset_type_counts)
  
  # Check Method-Dataset combinations in raw data
  cat("\nRaw Method-Dataset combinations:\n")
  raw_combo_counts <- scores_raw %>% 
    count(Method, Dataset) %>%
    arrange(Method, Dataset)
  cat("Total Method-Dataset combinations:", nrow(raw_combo_counts), "\n")
  cat("First 10 combinations:\n")
  print(head(raw_combo_counts, 10))
  
  # Specifically check CANDIES in raw data
  cat("\nCANDIES in raw data:\n")
  candies_raw <- scores_raw %>% 
    filter(Method == "CANDIES") %>%
    select(Method, Dataset, Dataset_Type, SC_Score, BioC_Score)
  print(candies_raw)
  
  cat("=== PROCESSING DATA ===\n")
  
  # Process and clean the data
  scores <- scores_raw %>%
    # Remove rows with missing essential data
    filter(
      !is.na(SC_Score),
      !is.na(BioC_Score),
      !is.na(Total_Score)
    ) %>%
    # Create standardized dataset names
    mutate(
      Dataset_Clean = case_when(
        Dataset == "HLN" ~ "Human Lymph Nodes",
        Dataset == "HT" ~ "Human Tonsils", 
        Dataset == "MISAR_S1" ~ "Mouse Embryos S1",
        Dataset == "MISAR_S2" ~ "Mouse Embryos S2",
        Dataset == "Mouse_Brain" ~ "Mouse Brain",
        Dataset == "Mouse_Spleen" ~ "Mouse Spleen",
        Dataset == "Mouse_Thymus" ~ "Mouse Thymus",
        TRUE ~ Dataset
      ),
      # Create output type and features (simplified for spatial omics)
      Output = "embed",  # Most spatial omics methods output embeddings
      Features = "FULL", # Using full feature space
      OutputFeatures = "embed_FULL",
      Scaling = "scaled",
      # Rename scores to match scib format
      `Spatial Coherence` = SC_Score,
      `Bio Conservation` = BioC_Score,
      `Overall Score` = Total_Score,
      Clustering_Method = clustering,
      # Add data type information
      Datatype = case_when(
        str_detect(Dataset_Type, "RNA_ADT") ~ "RNA+ADT",
        str_detect(Dataset_Type, "RNA_ATAC") ~ "RNA+ATAC", 
        TRUE ~ Dataset_Type
      )
    ) %>%
    # Select relevant columns
    select(
      Dataset = Dataset_Clean,
      Method,
      Datatype,
      Output,
      Features,
      OutputFeatures, 
      Scaling,
      `Overall Score`,
      `Spatial Coherence`,
      `Bio Conservation`,
      Clustering_Method,
      GT_Available
    ) %>%
    # Filter out methods with insufficient data
    filter(!is.na(`Overall Score`))
  
  cat("After processing:\n")
  cat("  Processed rows:", nrow(scores), "\n")
  cat("  Unique methods:", length(unique(scores$Method)), "\n")
  cat("  Unique datasets:", length(unique(scores$Dataset)), "\n")
  cat("  Unique datatypes:", length(unique(scores$Datatype)), "\n")
  
  # Check final Method-Dataset combinations
  final_combo_counts <- scores %>% 
    count(Method, Dataset) %>%
    arrange(Method, Dataset)
  cat("Final Method-Dataset combinations:", nrow(final_combo_counts), "\n")
  
  # Check if we still have multiple datasets per method-datatype
  method_datatype_counts <- scores %>%
    count(Method, Datatype) %>%
    arrange(Method, Datatype)
  cat("\nFinal Method-Datatype counts (this is KEY!):\n")
  print(method_datatype_counts)
  
  cat("=== LOAD DATA DEBUG END ===\n\n")
  
  return(scores)
}

#' Plot SMOBench scatter plot
#' 
#' Adapted from scib plotSummaryScatter function for SMOBench spatial omics data
#' 
#' @param scores Processed scores tibble
#' @param methods_pal Named vector of colors for methods
#' 
#' @return A ggplot2 object
plotSMOBenchScatter <- function(scores, methods_pal) {
  
  # Calculate medians for reference lines
  medians <- scores %>%
    group_by(Dataset) %>%
    summarise(
      `Spatial Coherence` = median(`Spatial Coherence`, na.rm = TRUE),
      `Bio Conservation` = median(`Bio Conservation`, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Type = "Median")
  
  # Create the main plot
  p <- ggplot(scores) +
    aes(
      x = `Spatial Coherence`,
      y = `Bio Conservation`, 
      colour = Method,
      size = `Overall Score`,
      shape = Datatype
    ) +
    # Add median reference lines
    geom_hline(
      data = medians,
      aes(yintercept = `Bio Conservation`),
      linetype = "dashed", 
      colour = "blue", 
      alpha = 0.7
    ) +
    geom_vline(
      data = medians,
      aes(xintercept = `Spatial Coherence`),
      linetype = "dashed",
      colour = "blue",
      alpha = 0.7
    ) +
    # Main points
    geom_point(alpha = 0.8, stroke = 0.5) +
    # Customize scales
    scale_colour_manual(
      values = methods_pal,
      name = "Method"
    ) +
    scale_size_continuous(
      range = c(1, 4),
      name = "Overall Score",
      guide = guide_legend(override.aes = list(alpha = 1))
    ) +
    scale_shape_manual(
      values = c("RNA+ADT" = 16, "RNA+ATAC" = 17),
      name = "Data Type",
      guide = guide_legend(override.aes = list(size = 3))
    ) +
    # Fixed aspect ratio for fair comparison
    coord_fixed() +
    # Facet by dataset
    facet_wrap(~ Dataset) +
    # Labels
    labs(
      x = "Spatial Coherence Score", 
      y = "Biological Conservation Score",
      title = "SMOBench Vertical Integration Performance",
      subtitle = "Spatial Coherence vs Biological Conservation Trade-offs",
      caption = "Blue dashed lines show median performance per dataset"
    ) +
    # Theme
    theme_minimal() +
    theme(
      legend.position = "right",
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 10),
      strip.text = element_text(size = 9, face = "bold"),
      strip.background = element_rect(fill = "grey90", colour = "grey70"),
      panel.border = element_rect(fill = NA, colour = "grey70"),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      plot.caption = element_text(size = 8, colour = "grey60"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8)
    ) +
    # Customize legend layout
    guides(
      colour = guide_legend(
        title.position = "top",
        ncol = 1,
        order = 1
      ),
      shape = guide_legend(
        title.position = "top", 
        ncol = 1,
        order = 2
      ),
      size = guide_legend(
        title.position = "top",
        ncol = 1, 
        order = 3
      )
    )
  
  return(p)
}

#' Get SMOBench methods color palette
#' 
#' @return Named vector of colors for SMOBench methods
getSMOBenchMethodsPal <- function() {
  
  methods_pal <- c(
    "CANDIES" = "#E31A1C",      # Red
    "SpatialGlue" = "#1F78B4",  # Blue  
    "SpaMosaic" = "#33A02C",    # Green
    "COSMOS" = "#FF7F00",       # Orange
    "PRAGA" = "#6A3D9A",        # Purple
    "SpaMV" = "#FB9A99",        # Light red
    "PRESENT" = "#A6CEE3",      # Light blue
    "SpaMultiVAE" = "#B2DF8A",  # Light green
    "SpaFusion" = "#FDBF6F",    # Soft orange
    "SMOPCA" = "#1B9E77",       # Teal
    "SpaMI" = "#B15928",        # Brown
    "SpaBalance" = "#CAB2D6"    # Lavender
  )
  
  return(methods_pal)
}

#' Generate comparison summary across all clustering methods
#' 
#' @param results_dir Path to results directory
#' @param out_dir Output directory
generateClusteringComparison <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                                       out_dir = "Results/plots/") {
  
  clustering_methods <- c("leiden", "louvain", "kmeans", "mclust") 
  all_scores <- tibble()
  
  # Load all clustering results
  for (clustering in clustering_methods) {
    scores <- loadSMOBenchScores(results_dir, clustering)
    if (nrow(scores) > 0) {
      all_scores <- bind_rows(all_scores, scores)
    }
  }
  
  if (nrow(all_scores) == 0) {
    cat("No data found for comparison plot\n")
    return()
  }
  
  # Create comparison plot
  comparison_plot <- all_scores %>%
    ggplot(aes(x = `Spatial Coherence`, y = `Bio Conservation`)) +
    geom_point(aes(colour = Method, shape = Datatype, size = `Overall Score`), 
               alpha = 0.7) +
    facet_grid(Clustering_Method ~ Dataset) +
    scale_colour_manual(values = getSMOBenchMethodsPal()) +
    scale_shape_manual(values = c("RNA+ADT" = 16, "RNA+ATAC" = 17)) +
    scale_size_continuous(range = c(0.5, 3)) +
    coord_fixed() +
    labs(
      x = "Spatial Coherence Score",
      y = "Biological Conservation Score", 
      title = "SMOBench Performance Comparison Across Clustering Methods",
      subtitle = "Rows: Clustering Methods, Columns: Datasets"
    ) +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 6),
      strip.text = element_text(size = 7),
      legend.position = "bottom",
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 7)
    ) +
    guides(
      colour = guide_legend(ncol = 4),
      shape = guide_legend(ncol = 2),
      size = guide_legend(ncol = 4)
    )
  
  # Save comparison plot
  filename <- glue("{out_dir}/SMOBench_clustering_comparison.pdf")
  
  ggsave(
    filename,
    comparison_plot,
    width = 420, height = 297, units = "mm",  # A3 landscape
    dpi = 300
  )
  
  cat("Saved clustering comparison plot:", filename, "\n")
}

#' Make Best Methods Scatter Plot
#' 
#' Creates aggregated scatter plot showing method performance averaged across datasets
#' with error bars showing variability. Shows all methods in a single plot without faceting.
#' 
#' @param results_dir Path to results directory
#' @param out_dir Output directory for plots
#' @param clustering_methods Vector of clustering methods to process
#' 
makeBestMethodsScatter <- function(results_dir = "Results/evaluation/vertical_integration/final_results/",
                                   out_dir = "Results/plots/",
                                   clustering_methods = c("leiden", "louvain", "kmeans", "mclust")) {
  
  # Create output directory if it doesn't exist
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  # Process each clustering method
  for (clustering in clustering_methods) {
    
    cat("Processing best methods scatter for clustering method:", clustering, "\n")
    
    # Load and process data
    scores <- loadSMOBenchScores(results_dir, clustering)
    
    if (nrow(scores) == 0) {
      cat("Warning: No data found for clustering method:", clustering, "\n")
      next
    }
    
    # Aggregate performance across datasets
    aggregated_scores <- aggregateScoresAcrossDatasets(scores)
    
    # Debug: Print aggregated scores to check error bar values
    cat("Sample of aggregated scores with error bars:\n")
    print(head(aggregated_scores[, c("Method", "Datatype", "Spatial Coherence", "Bio Conservation", "SC_SE", "BioC_SE")], 6))
    
    # Create three different plots
    
    # 1. Combined plot (RNA+ADT + RNA+ATAC)
    combined_plot <- plotBestMethodsScatter(aggregated_scores, getSMOBenchMethodsPal(), "Combined (RNA+ADT & RNA+ATAC)")
    
    # 2. RNA+ADT only plot
    rna_adt_scores <- aggregated_scores %>% filter(Datatype == "RNA+ADT")
    rna_adt_plot <- plotBestMethodsScatter(rna_adt_scores, getSMOBenchMethodsPal(), "RNA+ADT")
    
    # 3. RNA+ATAC only plot  
    rna_atac_scores <- aggregated_scores %>% filter(Datatype == "RNA+ATAC")
    rna_atac_plot <- plotBestMethodsScatter(rna_atac_scores, getSMOBenchMethodsPal(), "RNA+ATAC")
    
    # Save all plots in multiple formats
    plot_list <- list(
      "combined" = combined_plot,
      "rna_adt" = rna_adt_plot,
      "rna_atac" = rna_atac_plot
    )
    
    for (plot_name in names(plot_list)) {
      for (format in c("pdf", "png", "tiff")) {
        filename <- glue("{out_dir}/SMOBench_best_methods_{clustering}_{plot_name}.{format}")
        
        ggsave(
          filename,
          plot_list[[plot_name]],
          width = 210, height = 148, units = "mm",  # A5 landscape for cleaner single plot
          dpi = 300
        )
      }
      cat("Saved", plot_name, "plot for", clustering, "clustering\n")
    }
    
    cat("Saved best methods plots for", clustering, "clustering\n")
  }
  
  cat("SMOBench best methods scatter plots generation complete!\n")
}

#' Aggregate scores across datasets
#' 
#' Calculate mean and standard error for each method across all datasets
#' 
#' @param scores Raw scores tibble from loadSMOBenchScores
#' 
#' @return Aggregated scores tibble with mean and SE
aggregateScoresAcrossDatasets <- function(scores) {
  
  cat("=== AGGREGATION DEBUG START ===\n")
  cat("Input data structure:\n")
  cat("  Total rows:", nrow(scores), "\n")
  cat("  Unique methods:", paste(unique(scores$Method), collapse = ", "), "\n")
  cat("  Unique datasets:", paste(unique(scores$Dataset), collapse = ", "), "\n")
  cat("  Unique datatypes:", paste(unique(scores$Datatype), collapse = ", "), "\n")
  
  # Critical: Check Method-Datatype combinations count
  cat("\nMethod-Datatype combination counts:\n")
  combo_counts <- scores %>% 
    count(Method, Datatype) %>%
    arrange(Method, Datatype)
  print(combo_counts)
  
  # Focus on CANDIES to verify expected structure
  cat("\nCANDIES detailed breakdown:\n")
  candies_all <- scores %>% 
    filter(Method == "CANDIES") %>%
    select(Method, Dataset, Datatype, `Spatial Coherence`, `Bio Conservation`)
  print(candies_all)
  
  # Check datatype distribution
  cat("\nDatatype distribution:\n")
  datatype_summary <- scores %>%
    count(Datatype) %>%
    arrange(Datatype)
  print(datatype_summary)
  
  # Debug: Let's examine a specific method-datatype combination
  candies_rna_adt <- scores %>% 
    filter(Method == "CANDIES", Datatype == "RNA+ADT")
  cat("Debug: CANDIES RNA+ADT raw data:\n")
  print(candies_rna_adt[, c("Dataset", "Spatial Coherence", "Bio Conservation")])
  
  aggregated <- scores %>%
    # Group only by Method and Datatype to ensure multiple observations per group
    group_by(Method, Datatype) %>%
    summarise(
      # Calculate raw statistics first (using original column names)
      SC_mean = mean(`Spatial Coherence`, na.rm = TRUE),
      SC_sd = sd(`Spatial Coherence`, na.rm = TRUE),
      SC_n = sum(!is.na(`Spatial Coherence`)),
      
      BioC_mean = mean(`Bio Conservation`, na.rm = TRUE),
      BioC_sd = sd(`Bio Conservation`, na.rm = TRUE),
      BioC_n = sum(!is.na(`Bio Conservation`)),
      
      Overall_mean = mean(`Overall Score`, na.rm = TRUE),
      Overall_sd = sd(`Overall Score`, na.rm = TRUE),
      Overall_n = sum(!is.na(`Overall Score`)),
      
      # Additional info
      N_Datasets = n(),
      Dataset_List = paste(unique(Dataset), collapse = ", "),
      
      # Keep other variables (take first value since they should be the same)
      Output = first(Output),
      Features = first(Features),
      OutputFeatures = first(OutputFeatures),
      Scaling = first(Scaling),
      Clustering_Method = first(Clustering_Method),
      
      .groups = "drop"
    ) %>%
    # Calculate final columns and standard errors after grouping
    mutate(
      # Rename to expected column names
      `Spatial Coherence` = SC_mean,
      `Bio Conservation` = BioC_mean,
      `Overall Score` = Overall_mean,
      
      # Calculate standard errors
      SC_SE = ifelse(SC_n > 1, SC_sd / sqrt(SC_n), NA),
      BioC_SE = ifelse(BioC_n > 1, BioC_sd / sqrt(BioC_n), NA),
      Overall_SE = ifelse(Overall_n > 1, Overall_sd / sqrt(Overall_n), NA)
    )
    # No handling of invalid values - show raw calculation results
  
  # Debug output
  cat("Debug: Aggregated data sample with detailed SE info:\n")
  print(aggregated[1:3, c("Method", "Datatype", "N_Datasets", "SC_sd", "SC_n", "SC_SE", "BioC_SE")])
  
  return(aggregated)
}

#' Plot Best Methods Scatter
#' 
#' Creates aggregated scatter plot with error bars showing method performance
#' averaged across datasets
#' 
#' @param aggregated_scores Aggregated scores from aggregateScoresAcrossDatasets
#' @param methods_pal Named vector of colors for methods
#' @param subtitle_text Subtitle text for the plot
#' 
#' @return A ggplot2 object
plotBestMethodsScatter <- function(aggregated_scores, methods_pal, subtitle_text = "") {
  
  # Create the main plot
  p <- ggplot(aggregated_scores) +
    aes(
      x = `Spatial Coherence`,
      y = `Bio Conservation`,
      colour = Method,
      size = `Overall Score`,
      shape = Datatype
    ) +
    # Add error bars (scaled by factor 2.5 for emphasis)
    geom_errorbar(
      aes(
        ymin = `Bio Conservation` - BioC_SE * 2.5,
        ymax = `Bio Conservation` + BioC_SE * 2.5
      ),
      width = 0.02,
      alpha = 0.7,
      size = 0.5,
      show.legend = FALSE
    ) +
    geom_errorbarh(
      aes(
        xmin = `Spatial Coherence` - SC_SE * 2.5,
        xmax = `Spatial Coherence` + SC_SE * 2.5
      ),
      height = 0.02,
      alpha = 0.7,
      size = 0.5,
      show.legend = FALSE
    ) +
    # Main points
    geom_point(alpha = 0.8, stroke = 0.5) +
    # Add method labels
    geom_text(
      aes(label = Method),
      nudge_y = 0.03,
      size = 3,
      show.legend = FALSE,
      check_overlap = TRUE
    ) +
    # Customize scales
    scale_colour_manual(
      values = methods_pal,
      name = "Method"
    ) +
    scale_size_continuous(
      range = c(2, 6),
      name = "Overall Score",
      guide = guide_legend(override.aes = list(alpha = 1))
    ) +
    scale_shape_manual(
      values = c("RNA+ADT" = 16, "RNA+ATAC" = 17),
      name = "Data Type",
      guide = guide_legend(override.aes = list(size = 4))
    ) +
    # Set reasonable axis limits
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      labels = seq(0, 1, 0.2)
    ) +
    scale_y_continuous(
      limits = c(0, 1), 
      breaks = seq(0, 1, 0.2),
      labels = seq(0, 1, 0.2)
    ) +
    # Fixed aspect ratio for fair comparison
    coord_fixed() +
    # Labels
    labs(
      x = "Spatial Coherence Score (Mean ± SE)", 
      y = "Biological Conservation Score (Mean ± SE)",
      title = "SMOBench Method Performance Overview",
      subtitle = ifelse(subtitle_text != "", 
                       paste(subtitle_text, "- Aggregated performance across all datasets with error bars"),
                       "Aggregated performance across all datasets with error bars"),
      caption = "Error bars show standard error across datasets. Point size reflects overall score."
    ) +
    # Theme
    theme_minimal() +
    theme(
      legend.position = "right",
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12),
      panel.border = element_rect(fill = NA, colour = "grey70"),
      panel.grid.major = element_line(colour = "grey90", size = 0.5),
      panel.grid.minor = element_line(colour = "grey95", size = 0.25),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      plot.caption = element_text(size = 9, colour = "grey60"),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9)
    ) +
    # Customize legend layout
    guides(
      colour = guide_legend(
        title.position = "top",
        ncol = 1,
        order = 1,
        override.aes = list(size = 4)
      ),
      shape = guide_legend(
        title.position = "top", 
        ncol = 1,
        order = 2,
        override.aes = list(size = 4)
      ),
      size = guide_legend(
        title.position = "top",
        ncol = 1, 
        order = 3
      )
    )
  
  return(p)
}

# Main execution function
main <- function() {
  
  cat("=== SMOBench Summary Scatter Plot Generator ===\n")
  cat("Generating scatter plots for vertical integration results...\n\n")
  
  # Set paths (relative to SMOBench-CLEAN root)
  results_dir <- "Results/evaluation/vertical_integration/final_results/"
  out_dir <- "Results/plots/"
  
  # Check if results directory exists
  if (!dir.exists(results_dir)) {
    cat("Error: Results directory not found:", results_dir, "\n")
    cat("Please ensure you are running from SMOBench-CLEAN root directory\n")
    return()
  }
  
  # Generate individual clustering plots (Summary Scatter)
  cat("1. Generating Summary Scatter plots (faceted by dataset)...\n")
  makeSMOBenchScatter(results_dir, out_dir)
  
  # Generate best methods plots (Best Methods Scatter)
  cat("\n2. Generating Best Methods Scatter plots (aggregated across datasets)...\n")
  makeBestMethodsScatter(results_dir, out_dir)
  
  # Generate clustering comparison plot
  cat("\n3. Generating clustering comparison plot...\n")
  generateClusteringComparison(results_dir, out_dir)
  
  cat("\n=== SMOBench scatter plot generation complete! ===\n")
  cat("Check", out_dir, "for output files\n")
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
