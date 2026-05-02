"""Pydantic models for datasets, releases, tasks, and resolved outcomes.

The models are the canonical schema for the on-disk format. The
validation module wraps them with file-level checks (uniqueness across
rows, file presence, release/dataset directory layout).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ReleaseStatus = Literal["open", "closed", "archived"]


class ResolvedOutcome(BaseModel):
    """A resolution attached to a task after publish.

    `value` is always a list of outcome strings (single-outcome
    resolutions are wrapped: `["Yes"]`, never `"Yes"`). Each entry must
    appear in the parent task's `outcomes` and entries must be unique.
    """

    model_config = ConfigDict(extra="forbid")

    value: list[str] = Field(min_length=1)
    resolved_at: str | None = None
    source: str | None = None

    @field_validator("value")
    @classmethod
    def _check_value_entries(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("resolved_outcome.value entries must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError("resolved_outcome.value entries must be unique")
        return v


class Task(BaseModel):
    """One row of a release's `tasks.jsonl`.

    Required fields (`task_id`, `title`, `outcomes`) are frozen at
    publish. `resolved_outcome` is the only field expected to mutate
    after publish. All other fields are passed through unchanged.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str
    title: str
    outcomes: list[str] = Field(min_length=1)
    resolved_outcome: ResolvedOutcome | None = None

    @field_validator("task_id", "title")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("outcomes")
    @classmethod
    def _check_outcomes(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("outcomes entries must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _check_resolved_in_outcomes(self) -> Task:
        if self.resolved_outcome is not None:
            for entry in self.resolved_outcome.value:
                if entry not in self.outcomes:
                    raise ValueError(
                        f"resolved_outcome.value entry {entry!r} not in outcomes {self.outcomes!r}"
                    )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Dump the task back to a dict suitable for JSONL serialization."""
        return self.model_dump(exclude_none=True)


class ReleaseMetadata(BaseModel):
    """Schema for `release.json`."""

    model_config = ConfigDict(extra="ignore")

    release_id: str
    release_date: str
    description: str
    status: ReleaseStatus

    @field_validator("release_id", "release_date", "description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v


class DatasetMetadata(BaseModel):
    """Schema for `dataset.json`."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str

    @field_validator("name", "description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v
