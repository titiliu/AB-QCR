# AB-QCR

**Anchor–Bridge Quantile-Curve Recalibration** for day-ahead photovoltaic power forecasts.

Companion code for *Structured anchor–bridge recalibration of photovoltaic power quantile curves guided by pilot corrections*. AB-QCR constructs a bounded isotonic baseline, jointly fits the main anchor and local bridge corrections, and issues a bounded, noncrossing curve on q01–q99.

## Install and run

Use Python 3.11 or later. The portable float64 implementation uses NumPy and SciPy on CPU.

```bash
python -m pip install .
python examples/run_example.py
python -m unittest discover -s tests -v
```

The example generates forecasts in memory, selects a CAL layout with a small update budget, and checks prediction and model reload. The dependency versions used with Python 3.14 are listed in [requirements-tested.txt](requirements-tested.txt).

## Study data and reproducibility

The [raw-data release](https://github.com/titiliu/AB-QCR/releases/tag/raw-data-v1) contains the original SMARD Germany data for 2020–2023 and NIST Ground Mount data for 2015–2018. See the [reproduction guide](REPRODUCIBILITY.md) for archive checks, preprocessing, the upstream prediction requirements, and the study's CAL splits and final AB-QCR settings.

## Develop and fit within CAL

Provide per-unit upstream forecasts as an `(n, 99)` array in q01–q99 order, with aligned observations, lead times in minutes, and a Boolean daylight mask. Arrange the full CAL rows chronologically. Pass integer row indices for TRAIN, STOP, SELECT, and the warm and cold fitting windows. Cold DEV uses the SELECT rows. Apply the observation-availability cutoffs to each development split. To check those boundaries at runtime, supply `target_time`, `available_time`, and `issue_time` together.

```python
from abqcr import CALSplits, fit_cal, project_quantiles

splits = CALSplits(train, stop, select, warm_fit, warm_dev, cold_fit)
comparators = {**other_select_curves, "MONOTONE": project_quantiles(q_cal[select])}
model, development = fit_cal(q_cal, lead_minutes, y_cal, daylight, splits, comparators)
issued = model.predict(q_future, future_lead_minutes)
model.save("fitted.npz")
```

`other_select_curves` maps comparator names to their issued curves for the same SELECT records and in the same row order. Set it to an empty dictionary to use MONOTONE as the comparator.

1. Fit the coefficient candidates on TRAIN. Use daytime QS on STOP to retain eligible checkpoints and screen the regularization grid. Compare the retained configurations on SELECT.
2. For five data-driven anchors, propose interior positions separately in the warm and cold FIT windows. Use the two relative DEV scores to choose which sides to adopt.
3. Score the fixed three-anchor, fixed five-anchor, and data-driven five-anchor candidates on the common SELECT records. For each metric, rank each layout as one plus the number of comparators with a strictly lower score. Use the same comparators for every layout. Choose the layout with the lowest mean rank across QS, CRPS, CE, and MAE. Lower QS breaks a mean-rank tie. Exact remaining ties follow the listed layout order.
4. Refit the selected configuration on full CAL. Recompute the adopted pilot positions for a data-driven layout and retain the checkpoint with the lowest training objective. Use the fitted model to issue forecasts for the evaluation period.

Set `rungs` and `policy_updates` to control the portable solver's update budgets. Use `checkpoint_updates` to set the interior checkpoint schedule. `development` contains the selected layout, development scores, and full CAL fit information.

## Fit a specified layout from CSV

Use `abqcr fit` when the anchor positions and regularization have already been chosen.

```bash
abqcr fit --input CAL.csv --model fitted.npz --anchors 1 10 50 90 99 --updates 256
abqcr predict --input forecasts.csv --model fitted.npz --output calibrated.csv
abqcr score --predictions calibrated.csv --observations observations.csv
```

| CSV | Required columns |
| --- | --- |
| CAL | `record_id`, `lead_time_min`, `q01` through `q99`, `observed_pu` |
| Forecasts | `record_id`, `lead_time_min`, `q01` through `q99` |
| Observations for scoring | `record_id`, `observed_pu`, `daylight` |

Use unique record IDs. Scoring aligns observations by ID and uses the daylight mask. The JSON output contains QS, CRPS, CE, MAE, and the eligible record count. QS, CRPS, and MAE use p.u. CE uses percentage points.

## License and citation

The AB-QCR source code and repository documentation are available under the [MIT License](LICENSE). The original data retain their source terms. SMARD data are attributed to Bundesnetzagentur | SMARD.de under [CC BY 4.0](https://www.smard.de/en/datennutzung). The NIST archives are governed by the [NIST data license](https://www.nist.gov/open/license).

Use [CITATION.cff](CITATION.cff) to cite the software. Cite the original providers when using the data.
