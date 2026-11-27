"""Fit AB-QCR, issue calibrated curves, and score forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ABQCR
from .csvio import read_forecasts, read_observations, write_forecasts
from .metrics import score


def _distinct_output(destination: Path, *sources: Path) -> None:
    if any(destination.resolve() == source.resolve() for source in sources):
        raise ValueError("Choose an output path distinct from the input paths.")


def _fit(args: argparse.Namespace, **options) -> None:
    _distinct_output(args.model, args.input)
    records = read_forecasts(args.input, with_observations=True)
    model = ABQCR.fit(records.quantiles, records.lead_time_min, records.observed_pu, **options)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    print(json.dumps({
        "model": str(args.model), "anchors": list(model.anchors),
        "parameter_count": model.parameter_count,
        "fit": model.fit_summary,
    }, indent=2, allow_nan=False))


def _fit_custom(args: argparse.Namespace) -> None:
    _fit(
        args, anchors=tuple(args.anchors), time_knots=args.time_knots,
        lead_bounds=tuple(args.lead_bounds), regularization=tuple(args.regularization),
        updates=args.updates,
    )


def _predict(args: argparse.Namespace) -> None:
    _distinct_output(args.output, args.input, args.model)
    records = read_forecasts(args.input)
    model = ABQCR.load(args.model)
    predictions = model.predict(records.quantiles, records.lead_time_min)
    write_forecasts(args.output, records, predictions)
    print(f"Wrote {len(records.record_ids)} forecasts to {args.output}")


def _score(args: argparse.Namespace) -> None:
    records = read_forecasts(args.predictions)
    observations, mask = read_observations(
        args.observations, records.record_ids, all_rows=args.all_rows
    )
    print(json.dumps(score(records.quantiles, observations, daylight=mask), indent=2))


def _fit_paths(command: argparse.ArgumentParser) -> None:
    command.add_argument("--input", type=Path, required=True, help="CAL forecast CSV with observations")
    command.add_argument("--model", type=Path, required=True, help="Output model archive (.npz)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="Fit a specified AB-QCR layout and regularization")
    _fit_paths(fit)
    fit.add_argument("--anchors", type=int, nargs="+", default=(1, 10, 50, 90, 99))
    fit.add_argument("--time-knots", type=int, default=8)
    fit.add_argument("--lead-bounds", type=float, nargs=2, default=(720, 2145))
    fit.add_argument("--regularization", type=float, nargs=3, default=(1e-4, 1e-4, 1e-4))
    fit.add_argument("--updates", type=int, default=256)
    fit.set_defaults(run=_fit_custom)

    predict = commands.add_parser("predict", help="Issue calibrated q01–q99 curves")
    predict.add_argument("--input", type=Path, required=True, help="Upstream forecast CSV")
    predict.add_argument("--model", type=Path, required=True, help="Fitted AB-QCR model (.npz)")
    predict.add_argument("--output", type=Path, required=True, help="Issued forecast CSV")
    predict.set_defaults(run=_predict)

    evaluate = commands.add_parser(
        "score",
        help="Report QS, CRPS, CE, and MAE for issued curves",
        description="Report QS, CRPS, CE, MAE, and the eligible record count.",
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--observations", type=Path, required=True)
    evaluate.add_argument("--all-rows", action="store_true", help="Evaluate all rows instead of daylight")
    evaluate.set_defaults(run=_score)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"abqcr: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
