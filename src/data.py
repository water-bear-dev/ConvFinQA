"""
Load and index the cleaned ConvFinQA dataset.

The JSON file contains splits (e.g., train/dev) mapped to lists of records.
"""

from __future__ import annotations

import json

from .models import ConvFinQARecord


class DatasetLoader:
    """Loads convfinqa_dataset.json and provides O(1) record lookup by id."""

    def __init__(self, json_path: str):
        self.data: dict[str, list[ConvFinQARecord]] = {}
        self.id_map: dict[str, ConvFinQARecord] = {}
        self._load_data(json_path)
        self._build_id_map()

    def _load_data(self, json_path: str) -> None:
        with open(json_path) as f:
            raw_data = json.load(f)

        for split, records in raw_data.items():
            self.data[split] = [ConvFinQARecord(**rec) for rec in records]

    def _build_id_map(self) -> None:
        for records in self.data.values():
            for rec in records:
                self.id_map[rec.id] = rec

    def get_record(self, record_id: str) -> ConvFinQARecord | None:
        """Return a single record by id, or None if not found."""
        return self.id_map.get(record_id)

    def iter_split(self, split: str) -> list[ConvFinQARecord]:
        """Return all records in a split (e.g. 'dev')."""
        return self.data.get(split, [])

