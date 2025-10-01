"""Experiment orchestration for topic classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .data import DatasetSplits, iter_size_progression, load_medical_abstracts_dataset, sample_documents
from .model import MultinomialNaiveBayes
from .preprocess import TextPreprocessor, build_vocabulary, transform_to_bow


@dataclass
class ExperimentConfig:
    """Configuration for a supervised topic classification experiment."""

    training_sizes: Sequence[int | None]
    random_state: int = 42
    output_dir: Path = Path("reports")

    def output_path(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


@dataclass
class ExperimentResult:
    """Captured metrics for a single experiment run."""

    training_size: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "training_size": self.training_size,
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
        }


def _compute_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    label_names: Dict[int, str],
) -> Dict[str, float]:
    labels = sorted(label_names.keys())
    total = len(true_labels)
    correct = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth == pred)
    accuracy = correct / total if total else 0.0

    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for label in labels:
        tp = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    precision_macro = sum(precision_values) / len(precision_values) if precision_values else 0.0
    recall_macro = sum(recall_values) / len(recall_values) if recall_values else 0.0
    f1_macro = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def _classification_report(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    label_names: Dict[int, str],
) -> List[Dict[str, float | str]]:
    labels = sorted(label_names.keys())
    report: List[Dict[str, float | str]] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, predicted_labels) if truth == label and pred != label)
        support = sum(1 for truth in true_labels if truth == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        report.append(
            {
                "label": label_names.get(label, str(label)),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support,
            }
        )
    return report


def run_experiments(
    config: ExperimentConfig,
    dataset: DatasetSplits | None = None,
) -> List[ExperimentResult]:
    """Run a series of experiments over different training sizes."""

    if dataset is None:
        dataset = load_medical_abstracts_dataset()

    preprocessor = TextPreprocessor()
    train_tokens = [preprocessor(text) for text in dataset.train_texts]
    test_tokens = [preprocessor(text) for text in dataset.test_texts]

    vocabulary = build_vocabulary(train_tokens)
    train_vectors_full = transform_to_bow(train_tokens, vocabulary)
    test_vectors = transform_to_bow(test_tokens, vocabulary)

    results: List[ExperimentResult] = []
    size_iterator = list(iter_size_progression(config.training_sizes, max_size=len(train_vectors_full)))

    for requested_size in size_iterator:
        subset_vectors, subset_labels = sample_documents(
            train_vectors_full,
            dataset.train_labels,
            requested_size,
            random_state=config.random_state,
        )
        model = MultinomialNaiveBayes()
        model.fit(subset_vectors, subset_labels)
        predictions = model.predict(test_vectors)
        metrics = _compute_metrics(dataset.test_labels, predictions, dataset.label_names)
        results.append(
            ExperimentResult(
                training_size=len(subset_vectors),
                accuracy=metrics["accuracy"],
                precision_macro=metrics["precision_macro"],
                recall_macro=metrics["recall_macro"],
                f1_macro=metrics["f1_macro"],
            )
        )

    output_dir = config.output_path()
    metrics_path = output_dir / "experiment_metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as handle:
        header = "training_size,accuracy,precision_macro,recall_macro,f1_macro\n"
        handle.write(header)
        for result in results:
            row = result.to_dict()
            handle.write(
                f"{row['training_size']},{row['accuracy']},{row['precision_macro']},{row['recall_macro']},{row['f1_macro']}\n"
            )

    # Save detailed classification report for the final (largest) training size run.
    final_model = MultinomialNaiveBayes()
    final_model.fit(train_vectors_full, dataset.train_labels)
    final_predictions = final_model.predict(test_vectors)
    report = _classification_report(dataset.test_labels, final_predictions, dataset.label_names)
    report_path = output_dir / "classification_report.csv"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("label,precision,recall,f1_score,support\n")
        for entry in report:
            handle.write(
                f"{entry['label']},{entry['precision']},{entry['recall']},{entry['f1_score']},{entry['support']}\n"
            )

    return results

