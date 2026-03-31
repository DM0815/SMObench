#!/usr/bin/env Rscript

#' SMOBench Horizontal Integration Summary Table - Styled Visualization
#'
#' Generates scIB-style summary tables for horizontal integration results,
#' highlighting Spatial Coherence, Biological Conservation, and Batch Effect
#' Removal metrics alongside aggregate scores.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(stringr)
  library(purrr)
  library(ggnewscale)
  library(glue)
  library(fs)
})

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

HORIZONTAL_RESULTS_DIR <- "Results/evaluation/horizontal_integration"
OUTPUT_DIR <- "Results/horizontal_summary_table"
CLUSTERING_METHOD <- "leiden"

DATASET_INFO <- tribble(
  ~Dataset,       ~Datatype,   ~GT,    ~Group,
  "HLN",          "RNA+ADT",   TRUE,   "RNA_ADT_withGT",
  "HT",           "RNA+ADT",   TRUE,   "RNA_ADT_withGT",
  "Mouse_Thymus", "RNA+ADT",   FALSE,  "RNA_ADT_woGT",
  "Mouse_Spleen", "RNA+ADT",   FALSE,  "RNA_ADT_woGT",
  "MISAR_S1",     "RNA+ATAC",  TRUE,   "RNA_ATAC_withGT",
  "MISAR_S2",     "RNA+ATAC",  TRUE,   "RNA_ATAC_withGT",
  "Mouse_Brain",  "RNA+ATAC",  FALSE,  "RNA_ATAC_woGT"
)

BVC_WITH_GT <- c("ari", "nmi", "asw_celltype", "graph_clisi")
BVC_WO_GT   <- c("silhouette", "calinski_norm", "dbi_norm")
BER_METRICS <- c("kbet", "knn_connectivity", "basw", "ilisi", "pcr")

# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------

mean_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}

row_mean_safe <- function(df, cols) {
  if (length(cols) == 0) return(rep(NA_real_, nrow(df)))
  mat <- df[, cols, drop = FALSE]
  apply(mat, 1, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA_real_)
    mean(x)
  })
}

normalize_vector <- function(vec, lower_better = FALSE) {
  mask <- !is.na(vec)
  if (!any(mask)) return(rep(NA_real_, length(vec)))
  vals <- vec[mask]
  mn <- min(vals)
  mx <- max(vals)
  if (!is.finite(mn) || !is.finite(mx)) {
    out <- rep(NA_real_, length(vec))
    return(out)
  }
  if (mx == mn) {
    out <- rep(0.5, length(vec))
    out[!mask] <- NA_real_
    return(out)
  }
  out <- if (lower_better) {
    (mx - vec) / (mx - mn)
  } else {
    (vec - mn) / (mx - mn)
  }
  out[!mask] <- NA_real_
  pmin(pmax(out, 0), 1)
}

read_metric_value <- function(metrics_vec, key) {
  if (key %in% names(metrics_vec)) {
    return(as.numeric(metrics_vec[[key]]))
  }
  NA_real_
}

# -------------------------------------------------------------------------
# Data loading and aggregation
# -------------------------------------------------------------------------

load_horizontal_metrics <- function(base_dir = HORIZONTAL_RESULTS_DIR,
                                    clustering = CLUSTERING_METHOD) {
  method_dirs <- dir_ls(base_dir, type = "directory", recurse = FALSE)
  methods <- method_dirs %>%
    path_file() %>%
    setdiff("final_results") %>%
    sort()

  rows <- list()

  for (method in methods) {
    for (i in seq_len(nrow(DATASET_INFO))) {
      dataset <- DATASET_INFO$Dataset[i]
      suffix <- if (DATASET_INFO$GT[i]) "withGT" else "woGT"
      file_name <- glue("{method}_{dataset}_horizontal_{clustering}_{suffix}.csv")
      file_path <- path(base_dir, method, file_name)

      if (!file_exists(file_path)) {
        next
      }

      metrics_tbl <- read_csv(
        file_path,
        col_types = cols(
          Metric = col_character(),
          Value = col_double()
        ),
        na = c("", "NA")
      )

      metrics_vec <- metrics_tbl$Value
      names(metrics_vec) <- metrics_tbl$Metric

      rows[[length(rows) + 1]] <- tibble(
        Method = method,
        Dataset = dataset,
        Datatype = DATASET_INFO$Datatype[i],
        GT = DATASET_INFO$GT[i],
        moran = read_metric_value(metrics_vec, "Moran Index"),
        ari = read_metric_value(metrics_vec, "ARI"),
        nmi = read_metric_value(metrics_vec, "NMI"),
        asw_celltype = read_metric_value(metrics_vec, "asw_celltype"),
        graph_clisi = read_metric_value(metrics_vec, "graph_clisi"),
        dbi = read_metric_value(metrics_vec, "Davies-Bouldin Index"),
        silhouette = read_metric_value(metrics_vec, "Silhouette Coefficient"),
        calinski = read_metric_value(metrics_vec, "Calinski-Harabaz Index"),
        kbet = read_metric_value(metrics_vec, "kBET"),
        knn_connectivity = read_metric_value(metrics_vec, "KNN_connectivity"),
        basw = read_metric_value(metrics_vec, "bASW"),
        ilisi = read_metric_value(metrics_vec, "iLISI"),
        pcr = read_metric_value(metrics_vec, "PCR")
      )
    }
  }

  if (length(rows) == 0) {
    warning("No horizontal integration metrics found.")
    return(tibble())
  }

  metrics_df <- bind_rows(rows)

  if (nrow(metrics_df) == 0) {
    return(metrics_df)
  }

  metrics_df %>%
    group_by(Dataset) %>%
    mutate(
      dbi_norm = normalize_vector(dbi, lower_better = TRUE),
      calinski_norm = normalize_vector(calinski, lower_better = FALSE)
    ) %>%
    ungroup()
}

aggregate_horizontal_data <- function(metrics_df, datatype_filter, gt_filter) {
  subset_df <- metrics_df %>%
    filter(Datatype == datatype_filter, GT == gt_filter)

  if (nrow(subset_df) == 0) {
    message("Warning: No data for ", datatype_filter, " + GT=", gt_filter)
    return(NULL)
  }

  aggregated <- subset_df %>%
    group_by(Method) %>%
    summarise(
      moran = mean_na(moran),
      ari = mean_na(ari),
      nmi = mean_na(nmi),
      asw_celltype = mean_na(asw_celltype),
      graph_clisi = mean_na(graph_clisi),
      silhouette = mean_na(silhouette),
      calinski_norm = mean_na(calinski_norm),
      dbi_norm = mean_na(dbi_norm),
      kbet = mean_na(kbet),
      knn_connectivity = mean_na(knn_connectivity),
      basw = mean_na(basw),
      ilisi = mean_na(ilisi),
      pcr = mean_na(pcr),
      .groups = "drop"
    )

  if (nrow(aggregated) == 0) {
    return(NULL)
  }

  if (gt_filter) {
    bioc_cols <- intersect(BVC_WITH_GT, names(aggregated))
  } else {
    bioc_cols <- intersect(BVC_WO_GT, names(aggregated))
  }

  ber_cols <- intersect(BER_METRICS, names(aggregated))

  aggregated <- aggregated %>%
    mutate(
      spatial_score = moran,
      bio_score = row_mean_safe(pick(dplyr::all_of(bioc_cols)), bioc_cols),
      ber_score = row_mean_safe(pick(dplyr::all_of(ber_cols)), ber_cols),
      final_score = apply(
        cbind(spatial_score, bio_score, ber_score),
        1,
        function(x) {
          x <- x[!is.na(x)]
          if (length(x) == 0) return(NA_real_)
          mean(x)
        }
      )
    )

  if (gt_filter) {
    aggregated <- aggregated %>%
      transmute(
        Method,
        `Moran's Index` = moran,
        ARI = ari,
        NMI = nmi,
        `ASW celltype` = asw_celltype,
        `Graph cLISI` = graph_clisi,
        `kBET` = kbet,
        `KNN connectivity` = knn_connectivity,
        `bASW` = basw,
        `iLISI` = ilisi,
        `PCR` = pcr,
        `Spatial Coherence` = spatial_score,
        `Bio Conservation` = bio_score,
        `Batch Effect Removal` = ber_score,
        `Final Score` = final_score
      )
  } else {
    aggregated <- aggregated %>%
      transmute(
        Method,
        `Moran's Index` = moran,
        `Silhouette Coefficient` = silhouette,
        `Calinski Harabasz` = calinski_norm,
        `Davies Bouldin` = dbi_norm,
        `kBET` = kbet,
        `KNN connectivity` = knn_connectivity,
        `bASW` = basw,
        `iLISI` = ilisi,
        `PCR` = pcr,
        `Spatial Coherence` = spatial_score,
        `Bio Conservation` = bio_score,
        `Batch Effect Removal` = ber_score,
        `Final Score` = final_score
      )
  }

  aggregated %>%
    mutate(across(-Method, as.numeric)) %>%
    arrange(desc(`Final Score`))
}

# -------------------------------------------------------------------------
# Styled plot creation (adapted from vertical integration version)
# -------------------------------------------------------------------------

createStyledSummaryTable <- function(aggregated_data, title_suffix = "") {
  if (is.null(aggregated_data) || nrow(aggregated_data) == 0) {
    return(NULL)
  }

  col_order <- names(aggregated_data)
  plot_data <- aggregated_data %>%
    mutate(Method = factor(Method, levels = rev(Method))) %>%
    pivot_longer(-Method, names_to = "Metric", values_to = "Score") %>%
    mutate(
      x = as.numeric(factor(Metric, levels = col_order)),
      y = as.numeric(Method),
      section = case_when(
        Metric == "Moran's Index" ~ "Spatial Coherence",
        Metric %in% c("ARI", "NMI", "ASW celltype", "Graph cLISI",
                      "Silhouette Coefficient", "Calinski Harabasz", "Davies Bouldin") ~ "Biological Conservation",
        Metric %in% c("kBET", "KNN connectivity", "bASW", "iLISI", "PCR") ~ "Batch Effect Removal",
        Metric %in% c("Spatial Coherence", "Bio Conservation", "Batch Effect Removal", "Final Score") ~ "Aggregate Score",
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
      Score_clamped = ifelse(section == "Aggregate Score",
                             pmin(pmax(Score, 0), 1),
                             Score_scaled),
      color_value = Score_clamped
    )

  if (nrow(plot_data) == 0) {
    return(NULL)
  }

  section_bounds <- plot_data %>%
    filter(section != "Other") %>%
    distinct(Metric, section, x) %>%
    group_by(section) %>%
    summarise(
      xmin = min(x),
      xmax = max(x),
      xmid = (min(x) + max(x)) / 2,
      .groups = "drop"
    )

  header_data_top <- section_bounds %>%
    transmute(
      x = xmid + 0.1,
      y = nrow(aggregated_data) + 2.7,
      label = case_when(
        section == "Spatial Coherence" ~ "Spatial\nCoherence",
        section == "Biological Conservation" ~ "Biological\nConservation",
        section == "Batch Effect Removal" ~ "Batch Effect\nRemoval",
        section == "Aggregate Score" ~ "Aggregate\nScore",
        TRUE ~ section
      )
    )

  header_data_method <- tibble(
    x = 0.95,
    y = nrow(aggregated_data) + 1.15,
    label = "Method"
  )

  sub_metrics <- col_order[-1]
  header_data_sub <- tibble(
    x = 2:length(col_order),
    y = nrow(aggregated_data) + 1.15,
    label = case_when(
      sub_metrics == "Calinski Harabasz" ~ "Calinski\nHarabasz",
      sub_metrics == "Davies Bouldin" ~ "Davies\nBouldin",
      sub_metrics == "Batch Effect Removal" ~ "Batch Effect\nRemoval",
      TRUE ~ str_wrap(sub_metrics, width = 8)
    )
  )

  line_data_full <- tibble(
    x = 0.5, xend = length(col_order) + 0.5,
    y = nrow(aggregated_data) + 0.5, yend = nrow(aggregated_data) + 0.5
  )

  line_data_sections <- section_bounds %>%
    mutate(
      x = xmin - 0.5,
      xend = xmax + 0.5,
      y = nrow(aggregated_data) + 2.2,
      yend = nrow(aggregated_data) + 2.2
    )

  p <- ggplot() +
    geom_point(
      data = plot_data %>% filter(section == "Spatial Coherence"),
      aes(x = x, y = y, fill = color_value),
      shape = 21, size = 12, color = "white", stroke = 0.5
    ) +
    scale_fill_gradient(low = "#fdd49e", high = "#d94801", limits = c(0, 1)) +
    new_scale_fill() +

    geom_point(
      data = plot_data %>% filter(section == "Biological Conservation"),
      aes(x = x, y = y, fill = color_value),
      shape = 21, size = 12, color = "white", stroke = 0.5
    ) +
    scale_fill_gradient(low = "#c7e9c0", high = "#238b45", limits = c(0, 1)) +
    new_scale_fill() +

    geom_point(
      data = plot_data %>% filter(section == "Batch Effect Removal"),
      aes(x = x, y = y, fill = color_value),
      shape = 21, size = 12, color = "white", stroke = 0.5
    ) +
    scale_fill_gradient(low = "#c6dbef", high = "#08519c", limits = c(0, 1)) +
    new_scale_fill() +

    geom_tile(
      data = plot_data %>% filter(section == "Aggregate Score"),
      aes(x = x - (1 - Score_clamped) / 2, y = y, width = Score_clamped, height = 0.55, fill = color_value),
      color = "white", linewidth = 0.5
    ) +
    scale_fill_gradient(low = "#dadaeb", high = "#6a51a3", limits = c(0, 1)) +

    geom_text(
      data = plot_data %>% filter(section != "Aggregate Score"),
      aes(x = x, y = y, label = sprintf("%.2f", Score)),
      size = 3.3
    ) +
    geom_text(
      data = plot_data %>% filter(section == "Aggregate Score"),
      aes(x = x - 0.4, y = y, label = sprintf("%.2f", Score)),
      size = 3.3, hjust = 0
    ) +

    geom_text(
      data = header_data_top,
      aes(x = x, y = y, label = label),
      size = 5, fontface = "bold"
    ) +
    geom_text(
      data = header_data_method,
      aes(x = x, y = y, label = label),
      size = 3.8
    ) +
    geom_text(
      data = header_data_sub,
      aes(x = x, y = y, label = label),
      size = 3.6, lineheight = 0.9
    ) +

    geom_segment(
      data = line_data_full,
      aes(x = x, xend = xend, y = y, yend = yend)
    ) +
    geom_segment(
      data = line_data_sections,
      aes(x = x, xend = xend, y = y, yend = yend)
    ) +

    scale_y_continuous(
      breaks = 1:nrow(aggregated_data),
      labels = levels(plot_data$Method),
      expand = c(0, 0)
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    coord_cartesian(
      xlim = c(0.5, length(col_order) + 0.5),
      ylim = c(0.5, nrow(aggregated_data) + 3.5),
      clip = "off"
    ) +
    theme_void() +
    theme(
      axis.text.y = element_text(hjust = 0.5, face = "bold", margin = margin(r = -70)),
      legend.position = "none",
      plot.margin = margin(20, 20, 20, 20),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    ) +
    labs(title = paste("SMOBench Horizontal Integration Summary", title_suffix))

  p
}

# -------------------------------------------------------------------------
# Main generation routine
# -------------------------------------------------------------------------

generateHorizontalSummaryTables <- function(results_dir = HORIZONTAL_RESULTS_DIR,
                                            output_dir = OUTPUT_DIR,
                                            clustering_method = CLUSTERING_METHOD) {
  if (!dir_exists(output_dir)) {
    dir_create(output_dir, recurse = TRUE)
  }

  metrics_df <- load_horizontal_metrics(results_dir, clustering_method)

  if (nrow(metrics_df) == 0) {
    stop("No horizontal integration metrics available. Please run evaluation first.")
  }

  combinations <- list(
    list(datatype = "RNA+ADT", gt = TRUE,  name = "RNA_ADT_withGT", title = "RNA+ADT withGT"),
    list(datatype = "RNA+ADT", gt = FALSE, name = "RNA_ADT_woGT",  title = "RNA+ADT woGT"),
    list(datatype = "RNA+ATAC", gt = TRUE,  name = "RNA_ATAC_withGT", title = "RNA+ATAC withGT"),
    list(datatype = "RNA+ATAC", gt = FALSE, name = "RNA_ATAC_woGT",  title = "RNA+ATAC woGT")
  )

  for (combo in combinations) {
    message("\nGenerating summary for ", combo$name, "...")
    aggregated <- aggregate_horizontal_data(metrics_df, combo$datatype, combo$gt)

    if (is.null(aggregated) || nrow(aggregated) == 0) {
      message("  Skipped (no data).")
      next
    }

    plot_title <- paste(combo$title, "- Leiden")
    styled_plot <- createStyledSummaryTable(aggregated, plot_title)

    if (is.null(styled_plot)) {
      message("  Failed to create plot for ", combo$name)
      next
    }

    file_base <- file.path(output_dir, paste0("SMOBench_horizontal_styled_summary_", combo$name, "_leiden"))
    ggsave(paste0(file_base, ".png"), plot = styled_plot, width = 12, height = 7, dpi = 300, bg = "white")
    ggsave(paste0(file_base, ".pdf"), plot = styled_plot, width = 12, height = 7, bg = "white")

    message("  Saved summary table to: ", file_base, ".png/.pdf")
  }

  message("\nHorizontal integration styled summary tables complete.")
}

# -------------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------------

if (!interactive()) {
  generateHorizontalSummaryTables()
} else {
  message("Functions loaded. Call generateHorizontalSummaryTables() to create plots.")
}
