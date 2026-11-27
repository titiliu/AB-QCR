"""CSV input and output for quantile forecasts and observations."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

QUANTILE_COLUMNS = tuple(f"q{index:02d}" for index in range(1, 100))
FORECAST_COLUMNS = ("record_id", "lead_time_min", *QUANTILE_COLUMNS)


@dataclass(frozen=True)
class ForecastRecords:
    record_ids: tuple[str, ...]
    lead_time_min: np.ndarray
    quantiles: np.ndarray
    observed_pu: np.ndarray | None = None


def _require_columns(reader: csv.DictReader, required: tuple[str, ...]) -> None:
    header = reader.fieldnames or []
    if len(set(header)) != len(header):
        raise ValueError("CSV columns must be unique.")
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}.")


def _parse_id(row: dict, seen: set[str], line: int) -> str:
    record_id = (row.get("record_id") or "").strip()
    if not record_id or record_id in seen:
        raise ValueError(f"Line {line}: record_id must be nonempty and unique.")
    seen.add(record_id)
    return record_id


def _parse_number(row: dict, field: str, line: int) -> float:
    try:
        value = float(row[field])
    except (ValueError, TypeError) as error:
        raise ValueError(f"Line {line}: {field} must be numeric.") from error
    if not math.isfinite(value):
        raise ValueError(f"Line {line}: {field} must be finite.")
    return value


def _parse_daylight(value: str | None, line: int) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in ("1", "true"):
        return True
    if normalized in ("0", "false"):
        return False
    raise ValueError(f"Line {line}: daylight must be 0/1 or false/true.")


def read_forecasts(path: str | Path, *, with_observations: bool = False) -> ForecastRecords:
    """Read q01–q99 and forecast lead time in minutes, in CSV row order."""
    record_ids = []
    leads = []
    values = []
    targets = []
    seen: set[str] = set()
    required = (*FORECAST_COLUMNS, "observed_pu") if with_observations else FORECAST_COLUMNS
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader, required)
        for row in reader:
            line = reader.line_num
            record_ids.append(_parse_id(row, seen, line))
            leads.append(_parse_number(row, "lead_time_min", line))
            values.append([_parse_number(row, column, line) for column in QUANTILE_COLUMNS])
            if with_observations:
                targets.append(_parse_number(row, "observed_pu", line))
    if not record_ids:
        raise ValueError("Forecast CSV contains no records.")
    return ForecastRecords(
        tuple(record_ids),
        np.asarray(leads, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        np.asarray(targets, dtype=np.float64) if with_observations else None,
    )


def read_observations(
    path: str | Path, record_ids: tuple[str, ...], *, all_rows: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Join observed power and a daylight mask to predictions by unique record_id."""
    observations = {}
    seen: set[str] = set()
    required = ("record_id", "observed_pu")
    if not all_rows:
        required += ("daylight",)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader, required)
        for row in reader:
            line = reader.line_num
            record_id = _parse_id(row, seen, line)
            truth = _parse_number(row, "observed_pu", line)
            daylight = True if all_rows else _parse_daylight(row["daylight"], line)
            observations[record_id] = (truth, daylight)
    if set(record_ids) != seen:
        raise ValueError("Prediction and observation record_id sets differ.")
    ordered = np.asarray([observations[record_id] for record_id in record_ids])
    return ordered[:, 0], None if all_rows else ordered[:, 1].astype(bool)


def write_forecasts(
    path: str | Path, records: ForecastRecords, quantiles: np.ndarray
) -> None:
    """Write ordered 99-quantile rows with the original record IDs and leads."""
    if quantiles.shape != (len(records.record_ids), 99):
        raise ValueError("Predictions must have one q01–q99 row per record_id.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FORECAST_COLUMNS)
        for record_id, lead, values in zip(records.record_ids, records.lead_time_min, quantiles):
            writer.writerow((record_id, float(lead), *values.tolist()))
