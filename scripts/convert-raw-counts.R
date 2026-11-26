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

# The downloaded RDS is gzip-compressed twice, so readRDS needs a gzip connection.
connection <- gzcon(gzfile(raw_file, "rb"))
counts <- readRDS(connection)
close(connection)

cell_ids <- colnames(counts)
cell_samples <- sub("^[^_]+_", "", cell_ids)
# Select configured samples from count-matrix column names without using labels.
keep_cells <- cell_samples %in% selected_samples

# Build a sparse selected matrix without creating a second dense count matrix.
selected_columns <- which(keep_cells)
nonzero_counts <- vapply(
  selected_columns,
  function(index) sum(counts[[index]] != 0L),
  integer(1)
)
column_pointers <- c(0L, cumsum(nonzero_counts))
row_indices <- integer(column_pointers[length(column_pointers)])
values <- numeric(column_pointers[length(column_pointers)])

for (position in seq_along(selected_columns)) {
  column <- counts[[selected_columns[position]]]
  nonzero <- which(column != 0L)
  start <- column_pointers[position] + 1L
  end <- column_pointers[position + 1L]

  if (start <= end) {
    row_indices[start:end] <- nonzero
    values[start:end] <- column[nonzero]
  }
}

selected_counts <- sparseMatrix(
  i = row_indices,
  p = column_pointers,
  x = values,
  dims = c(nrow(counts), length(selected_columns))
)
gene_ids <- rownames(counts)
rm(counts, selected_columns, nonzero_counts, column_pointers, row_indices, values)
gc()

writeMM(selected_counts, output_files[1])
write.table(gene_ids, output_files[2],
            quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(cell_id = cell_ids[keep_cells],
                       sample = cell_samples[keep_cells]),
            output_files[3], quote = FALSE, row.names = FALSE, sep = "\t")
