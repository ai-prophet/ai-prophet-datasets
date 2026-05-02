"""Pydantic model tests — edge cases that the file-level validators rely on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_prophet_datasets.models import (
    DatasetMetadata,
    ReleaseMetadata,
    ResolvedOutcome,
    Task,
)


class TestResolvedOutcome:
    def test_minimal_single_value(self):
        ro = ResolvedOutcome(value=["Yes"])
        assert ro.value == ["Yes"]
        assert ro.resolved_at is None
        assert ro.source is None

    def test_multi_value(self):
        ro = ResolvedOutcome(value=["A", "C"])
        assert ro.value == ["A", "C"]

    def test_rejects_string_value(self):
        # The whole point of list-shape: never accept a bare string.
        with pytest.raises(ValidationError):
            ResolvedOutcome(value="Yes")  # type: ignore[arg-type]

    def test_rejects_empty_list(self):
        with pytest.raises(ValidationError):
            ResolvedOutcome(value=[])

    def test_rejects_duplicate_entries(self):
        with pytest.raises(ValidationError):
            ResolvedOutcome(value=["Yes", "Yes"])

    def test_rejects_empty_string_entry(self):
        with pytest.raises(ValidationError):
            ResolvedOutcome(value=["Yes", ""])

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ResolvedOutcome.model_validate({"value": ["Yes"], "weird": True})


class TestTask:
    def test_minimal(self):
        t = Task(task_id="t1", title="Q?", outcomes=["A", "B"])
        assert t.resolved_outcome is None

    def test_with_resolution(self):
        t = Task.model_validate(
            {
                "task_id": "t1",
                "title": "Q?",
                "outcomes": ["A", "B"],
                "resolved_outcome": {"value": ["A"]},
            }
        )
        assert t.resolved_outcome is not None
        assert t.resolved_outcome.value == ["A"]

    def test_resolution_value_must_be_in_outcomes(self):
        with pytest.raises(ValidationError, match="not in outcomes"):
            Task.model_validate(
                {
                    "task_id": "t1",
                    "title": "Q?",
                    "outcomes": ["A", "B"],
                    "resolved_outcome": {"value": ["C"]},
                }
            )

    def test_multi_resolution_partial_membership_fails(self):
        # Even one out-of-set entry is a hard failure.
        with pytest.raises(ValidationError, match="not in outcomes"):
            Task.model_validate(
                {
                    "task_id": "t1",
                    "title": "Q?",
                    "outcomes": ["A", "B"],
                    "resolved_outcome": {"value": ["A", "Z"]},
                }
            )

    def test_extra_fields_pass_through(self):
        t = Task.model_validate(
            {
                "task_id": "t1",
                "title": "Q?",
                "outcomes": ["A", "B"],
                "context": "background info",
                "metadata": {"category": "Sports"},
            }
        )
        assert t.model_dump()["context"] == "background info"
        assert t.model_dump()["metadata"] == {"category": "Sports"}

    def test_outcomes_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            Task(task_id="t1", title="Q?", outcomes=[])

    def test_task_id_cannot_be_blank(self):
        with pytest.raises(ValidationError):
            Task(task_id="   ", title="Q?", outcomes=["A"])


class TestReleaseAndDatasetMetadata:
    def test_release_status_enum(self):
        for s in ["open", "closed", "archived"]:
            ReleaseMetadata(release_id="r", release_date="2026-01-01", description="d", status=s)
        with pytest.raises(ValidationError):
            ReleaseMetadata(
                release_id="r", release_date="2026-01-01", description="d", status="weird"
            )

    def test_dataset_metadata_rejects_blank_name(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(name="", description="x")
