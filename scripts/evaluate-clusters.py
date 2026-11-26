import os

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)


CONFIG_FILE = "configs/default.yml"
ANNOTATION_FILE = "data/raw/GSE131907_Lung_Cancer_cell_annotation.txt.gz"
ASSIGNMENT_FILE = "outputs/clusters/clustering-labels.csv.gz"
ANNOTATION_LABELS_FILE = "outputs/annotations/annotation-labels.csv.gz"
SUMMARY_FILE = "outputs/tables/clustering-summary.csv"
ANNOTATION_EVALUATION_FILE = "outputs/tables/annotation-evaluation.csv"
PER_TYPE_FILE = "outputs/tables/annotation-per-type-evaluation.csv"
CONFUSION_FILE = "outputs/tables/annotation-confusion-matrix.csv"


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)
    with open(config["annotation"]["marker_panel_file"]) as file:
        marker_config = yaml.safe_load(file)

    os.makedirs("outputs/tables", exist_ok=True)
    assignments = pd.read_csv(ASSIGNMENT_FILE, index_col="cell_id")
    annotation_labels = pd.read_csv(ANNOTATION_LABELS_FILE, index_col="cell_id")
    annotation_labels = annotation_labels.loc[assignments.index, assignments.columns]
    annotations = pd.read_csv(ANNOTATION_FILE, sep=chr(9))
    annotations["cell_id"] = (
        annotations["Barcode"].astype(str) + "_" + annotations["Sample"].astype(str)
    )
    labels = annotations.set_index("cell_id").reindex(assignments.index)["Cell_type"]

    if labels.isna().any():
        raise ValueError("Reference Cell_type labels do not cover every clustered cell.")

    clustering_summary = pd.read_csv(SUMMARY_FILE)
    reference_order = marker_config["evaluation"]["source_label_order"]
    label_mapping = marker_config["evaluation"]["detailed_to_source"]
    annotation_evaluation_rows = []
    per_type_rows = []
    confusion_rows = []

    for column in assignments.columns:
        representation, method = column.rsplit("_", 1)
        summary = clustering_summary.loc[
            (clustering_summary["representation"] == representation)
            & (clustering_summary["method"] == method)
        ].iloc[0]
        # Apply the broad-label crosswalk only for the external comparison.
        detailed_labels = annotation_labels[column]
        predicted = detailed_labels.map(label_mapping)
        if predicted.isna().any():
            raise ValueError("Every annotation label must have an evaluation mapping.")
        combination = f"{representation}_{method}"
        macro_precision, _, _, _ = precision_recall_fscore_support(
            labels,
            predicted,
            labels=reference_order,
            average="macro",
            zero_division=0
        )
        annotation_evaluation_rows.append({
            "representation": representation,
            "method": method,
            "n_clusters": int(summary["n_clusters"]),
            "precision": macro_precision,
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1": f1_score(
                labels,
                predicted,
                labels=reference_order,
                average="macro",
                zero_division=0
            )
        })

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted,
            labels=reference_order,
            zero_division=0
        )
        per_type_rows.append(pd.DataFrame({
            "representation": representation,
            "method": method,
            "cell_type": reference_order,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }))

        predicted_order = reference_order + ["unmatched"]
        confusion = pd.crosstab(labels, predicted).reindex(
            index=reference_order,
            columns=predicted_order,
            fill_value=0
        )
        confusion.index.name = "reference_cell_type"
        confusion = confusion.stack().rename("cells").reset_index()
        confusion.columns = ["reference_cell_type", "predicted_label", "cells"]
        confusion.insert(0, "combination", combination)
        confusion.insert(0, "method", method)
        confusion.insert(0, "representation", representation)
        confusion_rows.append(confusion)

    pd.DataFrame(annotation_evaluation_rows).to_csv(
        ANNOTATION_EVALUATION_FILE,
        index=False
    )
    pd.concat(per_type_rows, ignore_index=True).to_csv(PER_TYPE_FILE, index=False)
    pd.concat(confusion_rows, ignore_index=True).to_csv(CONFUSION_FILE, index=False)
    print(f"Saved {ANNOTATION_EVALUATION_FILE}")
    print(f"Saved {PER_TYPE_FILE}")
    print(f"Saved {CONFUSION_FILE}")


if __name__ == "__main__":
    main()
