#!/usr/bin/env Rscript
# Overall integration summary table (vertical + horizontal)
#
# Usage:
#   conda run -n smobench bash -c "\
#     export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; \
#     export NUMBA_DISABLE_JIT=1; \
#     Rscript Draw/plot_integration_overall_summary.R"

suppressed_packages <- c("tidyverse", "scales")
for (pkg in suppressed_packages) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
if (length(script_path) == 0) {
  script_path <- file.path(getwd(), "Draw", "plot_integration_overall_summary.R")
}
script_dir <- dirname(normalizePath(script_path))
root_dir <- normalizePath(file.path(script_dir, ".."))
summary_csv <- file.path(root_dir, "Results", "summary_table", "method_level_scores.csv")
plot_dir <- file.path(root_dir, "Results", "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(plot_dir, "integration_overall_summary.png")

if (!file.exists(summary_csv)) {
  stop("Summary table not found: ", summary_csv)
}

data <- readr::read_csv(summary_csv, show_col_types = FALSE) |>
  dplyr::filter(Method != "SMOPCA")

metric_map <- tibble::tribble(
  ~Metric,            ~Label,         ~Group,       ~Section,      ~HeaderText,
  "Method",          "Method",      "Method",    "Method",     "Methods",
  "SC_Vertical",     "V_SC",        "Vertical",  "Vertical",   "Spatial\nCoherence",
  "BioC_Vertical",   "V_BioC",      "Vertical",  "Vertical",   "Biological\nConservation",
  "Final_Vertical",  "V_Final",     "Vertical",  "Vertical",   "Final\nScore",
  "SC_Horizontal",   "H_SC",        "Horizontal","Horizontal", "Spatial\nCoherence",
  "BioC_Horizontal", "H_BioC",      "Horizontal","Horizontal", "Biological\nConservation",
  "BER_Horizontal",  "H_BER",       "Horizontal","Horizontal", "Batch Effect\nRemoval",
  "Final_Horizontal", "H_Final",    "Horizontal","Horizontal", "Final\nScore"
) |>
  dplyr::mutate(col_id = dplyr::row_number())

if (!all(metric_map$Metric %in% colnames(data))) {
  missing_cols <- setdiff(metric_map$Metric, colnames(data))
  stop("Missing expected columns in summary table: ", paste(missing_cols, collapse = ", "))
}

final_cols <- dplyr::select(data, dplyr::starts_with("Final"))
ordered_methods <- data |>
  dplyr::mutate(
    Overall = rowMeans(as.matrix(final_cols), na.rm = TRUE)
  ) |>
  dplyr::arrange(dplyr::desc(Overall)) |>
  dplyr::pull(Method)

long_df <- data |>
  tidyr::pivot_longer(
    cols = metric_map$Metric[metric_map$Metric != "Method"],
    names_to = "Metric",
    values_to = "Value"
  ) |>
  dplyr::left_join(metric_map, by = "Metric") |>
  dplyr::mutate(
    Method = factor(Method, levels = ordered_methods),
    Label = factor(Label, levels = metric_map$Label[metric_map$Metric != "Method"])
  )

method_levels_plot <- rev(levels(long_df$Method))
method_col_id <- metric_map$col_id[metric_map$Metric == "Method"]
metric_cols <- metric_map |>
  dplyr::filter(Metric != "Method")
vertical_col_ids <- metric_cols$col_id[metric_cols$Group == "Vertical"]
horizontal_col_ids <- metric_cols$col_id[metric_cols$Group == "Horizontal"]

background_df <- tidyr::expand_grid(
  Method = method_levels_plot,
  col_id = metric_map$col_id
) |>
  dplyr::left_join(
    metric_map |> dplyr::select(col_id, Label, Group),
    by = "col_id",
    suffix = c("", ".map")
  ) |>
  dplyr::mutate(col_id = as.numeric(col_id)) |>
  dplyr::mutate(
    row_id = as.numeric(factor(Method, levels = method_levels_plot)),
    xmin = col_id - 0.45,
    xmax = col_id + 0.45,
    ymin = row_id - 0.45,
    ymax = row_id + 0.45,
    fill_color = "#ffffff"
  )

plot_df <- long_df |>
  dplyr::filter(!is.na(Value)) |>
  dplyr::mutate(
    row_id = as.numeric(factor(Method, levels = method_levels_plot)),
    xmin = col_id - 0.45,
    xmax = xmin + 0.85 * pmin(pmax(Value, 0), 1),
    ymin = row_id - 0.35,
    ymax = row_id + 0.35,
    text_x = col_id - 0.43,
    base_color = ifelse(Group == "Vertical", "#ff7f0e", "#9467bd"),
    bar_fill = alpha(base_color, 0.25 + 0.75 * pmin(pmax(Value, 0), 1))
  )

method_df <- tibble::tibble(
  Method = method_levels_plot,
  row_id = as.numeric(factor(Method, levels = method_levels_plot)),
  col_id = method_col_id
)

top_headers <- tibble::tibble(
  label = c("Methods", "Vertical Integration", "Horizontal Integration"),
  xmin = c(
    method_col_id - 0.45,
    min(vertical_col_ids) - 0.45,
    min(horizontal_col_ids) - 0.45
  ),
  xmax = c(
    method_col_id + 0.45,
    max(vertical_col_ids) + 0.45,
    max(horizontal_col_ids) + 0.45
  )
)

sub_headers <- dplyr::bind_rows(
  tibble::tibble(col_id = method_col_id, label = "Methods"),
  metric_cols |> dplyr::select(col_id, label = HeaderText)
) |>
  dplyr::arrange(col_id) |>
  dplyr::mutate(
    fontface = ifelse(col_id == method_col_id, "bold", "plain")
  )

p <- ggplot() +
  geom_rect(
    data = background_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill_color),
    color = "white"
  ) +
  geom_rect(
    data = plot_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = bar_fill),
    color = NA
  ) +
  geom_text(
    data = plot_df,
    aes(x = text_x, y = row_id, label = sprintf("%.2f", Value)),
    hjust = 0,
    size = 3.2
  ) +
  geom_text(
    data = method_df,
    aes(x = col_id, y = row_id, label = Method),
    fontface = "bold",
    size = 3.5,
    hjust = 0.5
  ) +
  scale_fill_identity() +
  coord_cartesian(
    xlim = c(0.5, max(metric_map$col_id) + 0.5),
    ylim = c(0.5, length(method_levels_plot) + 3),
    clip = "off"
  ) +
  theme_void() +
  theme(
    plot.margin = margin(45, 20, 25, 80),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  geom_text(
    data = top_headers,
    aes(x = (xmin + xmax) / 2, y = length(method_levels_plot) + 2.3, label = label),
    fontface = "bold",
    size = 4.2
  ) +
  geom_segment(
    data = top_headers |> dplyr::filter(label != "Methods"),
    aes(
      x = xmin,
      xend = xmax,
      y = length(method_levels_plot) + 2.1,
      yend = length(method_levels_plot) + 2.1
    ),
    linewidth = 0.6
  ) +
  geom_text(
    data = sub_headers,
    aes(x = col_id, y = length(method_levels_plot) + 1.6, label = label),
    size = 3.4,
    fontface = sub_headers$fontface,
    lineheight = 0.9
  ) +
  labs(title = "Integration Summary (Vertical + Horizontal)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))

method_labels <- method_levels_plot

ggsave(output_path, p, width = 11, height = max(4.5, 0.6 * length(method_levels_plot)), dpi = 300)

message("Saved integration summary figure: ", output_path)
