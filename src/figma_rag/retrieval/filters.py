"""Metadata filter helpers for retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CHROMA_OPERATORS = {
    "=": "$eq",
    "!=": "$ne",
    "<": "$lt",
    "<=": "$lte",
    ">": "$gt",
    ">=": "$gte",
}

OPERATOR_DESCRIPTIONS = {
    "$eq": "equals",
    "$ne": "not_equals",
    "$lt": "less_than",
    "$lte": "less_than_or_equal",
    "$gt": "greater_than",
    "$gte": "greater_than_or_equal",
}


@dataclass(frozen=True)
class MetadataFilter:
    """One constraint over a Chroma metadata field."""

    field: str
    operator: str
    value: str | int | float | bool

    def __post_init__(self) -> None:
        """Validate that the field and value can be used in a metadata query."""

        if not self.field.strip():
            raise ValueError("metadata filter field must not be empty")
        if self.operator not in OPERATOR_DESCRIPTIONS:
            raise ValueError(f"unsupported metadata filter operator: {self.operator}")
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("metadata filter value must not be empty")

    def to_chroma_clause(self) -> dict:
        """Return this filter as a single Chroma where clause."""

        return {self.field: {self.operator: self.value}}

    def to_description(self) -> dict[str, str | int | float | bool]:
        """Return a JSON-serializable description of the filter."""

        return {
            "field": self.field,
            "operator": OPERATOR_DESCRIPTIONS[self.operator],
            "value": self.value,
        }


@dataclass(frozen=True)
class MetadataFilterSet:
    """A set of metadata filters combined with AND semantics."""

    filters: tuple[MetadataFilter, ...] = ()

    @property
    def enabled(self) -> bool:
        """Return whether the filter set contains any filters."""

        return bool(self.filters)

    def to_chroma_where(self) -> dict | None:
        """Return the Chroma where payload for metadata filtering."""

        if not self.filters:
            return None

        clauses = [metadata_filter.to_chroma_clause() for metadata_filter in self.filters]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def to_description(self, enabled: bool = True) -> dict:
        """Return a JSON-serializable description for result artifacts."""

        return {
            "enabled": enabled,
            "combination": "and",
            "filters": [
                metadata_filter.to_description()
                for metadata_filter in self.filters
            ],
        }


def parse_metadata_filter(value: str) -> MetadataFilter:
    """Parse one command-line metadata filter.

    Supported forms include FIELD=VALUE, FIELD!=VALUE, FIELD<VALUE,
    FIELD<=VALUE, FIELD>VALUE, and FIELD>=VALUE.
    """

    for text_operator in (">=", "<=", "!=", "=", ">", "<"):
        if text_operator not in value:
            continue

        field, filter_value = value.split(text_operator, maxsplit=1)
        operator = CHROMA_OPERATORS[text_operator]
        parsed_value = _parse_filter_value(filter_value.strip(), operator)
        return MetadataFilter(
            field=field.strip(),
            operator=operator,
            value=parsed_value,
        )

    raise ValueError(
        f"Invalid metadata filter {value!r}. Expected FIELD=VALUE, FIELD<VALUE, "
        "FIELD<=VALUE, FIELD>VALUE, FIELD>=VALUE, or FIELD!=VALUE."
    )


def parse_metadata_filter_set(values: Iterable[str] | None) -> MetadataFilterSet:
    """Parse command-line metadata filters into a validated filter set."""

    if values is None:
        return MetadataFilterSet()
    return MetadataFilterSet(tuple(parse_metadata_filter(value) for value in values))


def _parse_filter_value(value: str, operator: str) -> str | int | float | bool:
    """Parse a CLI filter value into a Chroma-compatible metadata scalar."""

    if not value:
        raise ValueError("metadata filter value must not be empty")

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if operator in {"$lt", "$lte", "$gt", "$gte"}:
        raise ValueError(
            f"metadata filter operator {operator} requires a numeric value, got {value!r}"
        )
    return value
