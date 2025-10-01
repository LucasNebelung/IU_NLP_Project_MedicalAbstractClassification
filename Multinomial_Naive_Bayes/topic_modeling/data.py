"""Dataset utilities for supervised topic classification experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_DEFAULT_DATA_DIR = Path("/teamspace/studios/this_studio/data/data_excluding_5")
_LEGACY_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_EXCLUDED_LABELS = {5}


@dataclass
class DatasetSplits:
    """Container for dataset splits used in experiments."""

    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    label_names: Dict[int, str]


def load_medical_abstracts_dataset(
    data_dir: Path | str = _DEFAULT_DATA_DIR,
) -> DatasetSplits:
    """Load the curated medical abstracts dataset packaged with the repository."""

    data_path = Path(data_dir)
    if not data_path.exists():
        if data_path == _DEFAULT_DATA_DIR and _LEGACY_DATA_DIR.exists():
            data_path = _LEGACY_DATA_DIR
        else:
            raise FileNotFoundError(
                "Medical abstracts dataset not found. Expected files under "
                f"{data_path}. Provide a valid 'data_dir' argument if the data "
                "is stored elsewhere."
            )
    labels = _load_label_mapping(data_path / "medical_tc_labels.csv")
    filtered_labels = {
        label: name for label, name in labels.items() if label not in _EXCLUDED_LABELS
    }

    remapped_labels, index_map = _normalize_label_space(filtered_labels)

    train_texts, train_labels_raw = _load_split(
        data_path / "medical_tc_train_raw_excl_5.csv",
        excluded_labels=_EXCLUDED_LABELS,
    )
    test_texts, test_labels_raw = _load_split(
        data_path / "medical_tc_test_raw_excl_5.csv",
        excluded_labels=_EXCLUDED_LABELS,
    )

    train_labels = [_remap_label(label, index_map) for label in train_labels_raw]
    test_labels = [_remap_label(label, index_map) for label in test_labels_raw]

    return DatasetSplits(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=remapped_labels,
    )


def _load_label_mapping(path: Path) -> Dict[int, str]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {int(row["condition_label"]): row["condition_name"] for row in reader}


def _load_split(
    path: Path,
    *,
    excluded_labels: Sequence[int] | None = None,
) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    excluded = set(excluded_labels or [])
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = int(row["condition_label"])
            if label in excluded:
                continue
            labels.append(label)
            texts.append(row["medical_abstract"].strip())
    return texts, labels



def _normalize_label_space(labels: Dict[int, str]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Re-index labels so they form a contiguous zero-based range."""

    sorted_labels = sorted(labels.items())
    index_map: Dict[int, int] = {}
    normalized: Dict[int, str] = {}

    for new_index, (original_label, name) in enumerate(sorted_labels):
        index_map[original_label] = new_index
        normalized[new_index] = name

    return normalized, index_map


def _remap_label(label: int, mapping: Dict[int, int]) -> int:
    try:
        return mapping[label]
    except KeyError as error:
        raise KeyError(f"Label {label!r} is not defined in the label mapping") from error



def sample_documents(
    documents: Sequence[str],
    labels: Sequence[int],
    sample_size: int | None,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:
    """Sample documents without replacement when a subset size is requested."""

    if sample_size is None or sample_size >= len(documents):
        return list(documents), list(labels)

    rng = random.Random(random_state)
    indices = list(range(len(documents)))
    rng.shuffle(indices)
    indices = indices[:sample_size]
    return [documents[i] for i in indices], [labels[i] for i in indices]


def iter_size_progression(
    sizes: Iterable[int | None],
    max_size: int,
) -> Iterable[int | None]:
    """Yield deduplicated, valid training sizes for an experiment progression."""

    seen = set()
    for size in sizes:
        normalized = None if size is None else min(size, max_size)
        marker = max_size if normalized is None else normalized
        if marker in seen:
            continue
        seen.add(marker)
        yield normalized

