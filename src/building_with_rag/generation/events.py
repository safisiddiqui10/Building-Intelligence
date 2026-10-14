"""Internal events of one generation operation. Not response models, not SSE events."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import GenerationResult


@dataclass
class AttemptStarted:
    attempt: int
    max_attempts: int


@dataclass
class Text:
    text: str


@dataclass
class CheckFailed:
    attempt: int
    check: str
    detail: str


@dataclass
class Retrying:
    attempt: int  # the attempt about to start
    max_attempts: int
    details: list[str] = field(default_factory=list)


@dataclass
class Final:
    result: GenerationResult


Event = AttemptStarted | Text | CheckFailed | Retrying | Final
