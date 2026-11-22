library(Matrix)
library(yaml)

config <- read_yaml("configs/default.yml")
selected_samples <- config$dataset$selected_samples

raw_file <- "data/raw/GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz"
output_files <- c(
  "data/processed/lung_tumor_raw_counts.mtx",
  "data/processed/lung_tumor_genes.tsv",
  "data/processed/lung_tumor_cells.tsv"
)

if (all(file.exists(output_files))) {
  cat("Converted files already exist in data/processed/\n")
  quit()
}

connection <- gzcon(gzfile(raw_file, "rb"))
counts <- readRDS(connection)
close(connection)

cell_ids <- colnames(counts)
cell_samples <- sub("^[^_]+_", "", cell_ids)
keep_cells <- cell_samples %in% selected_samples

selected_counts <- as.matrix(counts[, keep_cells, drop = FALSE])
rm(counts)
gc()

writeMM(Matrix(selected_counts, sparse = TRUE), output_files[1])
write.table(rownames(selected_counts), output_files[2],
            quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(cell_id = cell_ids[keep_cells],
                       sample = cell_samples[keep_cells]),
            output_files[3], quote = FALSE, row.names = FALSE, sep = "\t")
