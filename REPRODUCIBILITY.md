# Reproducing the AB-QCR calibration workflow

The public materials include the AB-QCR calibration code, a synthetic [example](examples/run_example.py), and the original data archives. The historical upstream forecast banks, upstream training scripts and checkpoints, comparator forecast banks, and original GPU calibration checkpoints are outside this release. The steps below describe how to prepare compatible inputs and run the portable CPU calibration with supplied forecasts. Reproducing the paper's numerical tables requires the corresponding historical forecast inputs and fitting artifacts.

## 1. Download the original data

Download the five archives and `SHA256SUMS.txt` from the [raw-data release](https://github.com/titiliu/AB-QCR/releases/tag/raw-data-v1). From the directory containing the downloads, run `sha256sum -c SHA256SUMS.txt`, or compare the SHA-256 digests with `Get-FileHash -Algorithm SHA256` in PowerShell. The release's [data source notes](https://github.com/titiliu/AB-QCR/releases/download/raw-data-v1/DATA-SOURCES.md) provide the file coverage and attribution.

| Dataset | Archive | Source terms |
| --- | --- | --- |
| Germany, 2020–2023 | `smard-germany-pv-2020-2023-source.zip` | Bundesnetzagentur \| SMARD.de, [CC BY 4.0](https://www.smard.de/en/datennutzung) |
| NIST Ground Mount, 2015–2018 | `onemin-Ground-2015.zip` through `onemin-Ground-2018.zip` | [NIST dataset DOI](https://doi.org/10.18434/M3S67G) and [NIST data license](https://www.nist.gov/open/license) |

The [MIT License](LICENSE) covers the AB-QCR software and associated repository documentation. Use the source terms above for the data.

## 2. Build the observation series

**SMARD Germany.** Read the `series` timestamp–energy pairs from the weekly JSON files for series 4068, region DE, at 15-minute resolution. Convert the Unix millisecond timestamps to UTC interval starts and retain the window from 2020-01-01 00:00 up to 2024-01-01 00:00, exclusive. Multiply each valid interval's net feed-in in MWh by four to obtain mean MW, then divide by the fixed 100,000 MW reference. Set observation availability to 75 minutes after the start of the interval. The daylight flag uses solar elevation at the target timestamp, with 51° N and 10° E as the reference coordinates.

**NIST Ground Mount.** Read `TIMESTAMP` and the AC meter power field `PwrMtrP_kW_Avg` in kW from the one-minute CSV files. Use fixed UTC−05:00 throughout the year. Treat nonfinite values and values outside [−5.42, 325.2] kW as missing, and set the remaining negative values to zero. Compute an endpoint-labeled, right-closed five-minute mean only when all five expected one-minute observations are valid. For maintenance screening, group the five-minute endpoints by source-clock date. Mark a date's five-minute power means as unavailable if the cleaned five-minute power is zero and the complete five-minute mean of `SEWSPOAIrrad_Wm2_Avg` exceeds 100 W/m² at six or more consecutive endpoints. At each quarter-hour endpoint, average the three valid means ending at that time and at 5 and 10 minutes before it, and normalize by 271 kW DC rated capacity. Set observation availability to that 15-minute endpoint. The daylight flag uses solar elevation at the endpoint, with 39.1319° N and 77.2141° W for the array.

Keep valid zero-output observations. Use `solar_elevation_deg > 0` as the daylight flag and preserve all valid day and night rows for fitting. Retain the observation, target timestamp, availability timestamp, daylight flag, and a stable record ID for each row.

## 3. Supply upstream forecasts

For each target day, issue forecasts at noon the previous day in the dataset's clock. The 96 target timestamps run from 00:00 to 23:45, with leads of 720 to 2145 minutes at 15-minute steps. Generate one row of q01–q99 forecasts in per-unit power for each issuance and lead, using historical power available at issuance, lead time, calendar features, and solar position.

The paper used QRF, XGBoost-QR, LightGBM-QR, QR-GRU, QR-LSTM, Transformer-QR, and iTransformer-QR. Use external implementations if retraining these upstreams, including a quantile regression forest, XGBoost and LightGBM quantile objectives, and quantile-output PyTorch models for the recurrent and Transformer networks. Choose upstream settings on validation data within the upstream training period. Refit on the full upstream training period, then export both CAL and EVAL forecasts on the same q01–q99 grid. To run the calibration on forecasts you already have, align those predictions with the observations by record ID or by issuance and target timestamp. Each dataset–upstream pair has its own calibration run.

## 4. Prepare CAL and EVAL

| Dataset clock | Upstream training | CAL | EVAL | Full-CAL cutoff | CAL rows | Daylight EVAL rows |
| --- | --- | --- | --- | --- | ---: | ---: |
| Germany, UTC | 2020–2021 | 2022 | 2023 | 2022-12-31 12:00 | 34,987 | 17,640 |
| NIST, fixed UTC−05:00 | 2015–2016 | 2017 | 2018 | 2017-12-31 12:00 | 34,232 | 16,915 |

Use nominal TRAIN, STOP and SELECT periods starting on January 1, October 1 and November 7 of the CAL year. At a phase boundary, retain earlier rows only when both their target and observation availability precede the first forecast issuance in the following phase. Apply the full-CAL cutoff to all CAL rows. Keep the valid full CAL rows in chronological order, including those removed from a development subset at an internal boundary.

Table A.1 of the paper gives the two-stage development windows for LightGBM-QR and iTransformer-QR in Germany. In UTC, warm FIT covers [2022-01-01, 2022-06-01) and warm DEV covers [2022-07-01, 2022-09-01). Cold FIT covers [2022-01-01, 2022-09-01), and cold DEV uses SELECT. In each FIT window, require target and availability timestamps to be strictly earlier than the corresponding cutoff, 12:00 UTC on May 31 for warm FIT and August 31 for cold FIT. For other applications, construct and supply the appropriate legal warm FIT, warm DEV, and cold FIT indices explicitly. The API uses SELECT as cold DEV.

## 5. Develop a layout and fit AB-QCR

After [installation](README.md#install-and-run), use the Python API. The argument arrays below are prepared by the caller. `q_cal` has shape `(n, 99)`. The lead, observation, daylight, and time arrays each have length `n` and follow the same full CAL row order. Store all three time arrays with the same time origin and units, such as `numpy.datetime64` in UTC. The split arrays are strictly increasing, zero-based CAL row indices. Each of STOP, SELECT, and warm DEV requires daylight observations.

```python
from abqcr import CALSplits, fit_cal, project_quantiles

splits = CALSplits(train, stop, select, warm_fit, warm_dev, cold_fit)
comparators = {**other_select_curves, "MONOTONE": project_quantiles(q_cal[select])}
model, report = fit_cal(
    q_cal, lead_minutes, y_cal, daylight, splits, comparators,
    target_time=target_time, available_time=available_time, issue_time=issue_time,
)
model.save("fitted.npz")
```

`other_select_curves` maps each external comparator to its bounded, noncrossing q01–q99 curves for the same SELECT IDs, in SELECT order. Use an empty mapping for a run with MONOTONE as the sole comparator. Use the same comparator set for every candidate. The comparison set affects the ranks and can affect the selected layout.

The API fits on TRAIN, uses daytime QS on STOP to select checkpoints and screen regularization, and uses daytime QS on SELECT to compare the retained regularization settings per branch. The data-driven branch compares four policies in warm and cold DEV. Each policy retains both preset interior anchors or adopts a proposed anchor on the left, the right, or both sides. On SELECT, each candidate layout's rank for a metric is one plus the number of common comparators with a strictly lower score. The fixed three-anchor, fixed five-anchor, or data-driven five-anchor layout with the lowest mean rank across QS, CRPS, CE, and MAE is selected. Lower QS breaks a mean-rank tie, followed by the listed order for any exact remaining ties. The winner is refitted on full CAL. The warm and cold policy checkpoints determine the full CAL update target for the data-driven branch, and the adopted anchor positions are recomputed on full CAL. The model stores the full CAL checkpoint with the lowest training objective.

Record the code commit, the input digests, `report["schedule"]`, `report["split_records"]`, `report["selected_layout"]`, and `report["full_cal"]` with the saved model. The portable solver uses update budgets and a checkpoint grid. The study used a GPU solver with both update targets and active-time limits, as described in Appendix A.3.

## 6. Use a specified fit and evaluate

Appendix B of the paper lists the historical full CAL configurations, reproduced here for reference. Anchors are q-indices. The three regularization values are for temporal smoothing, state smoothing, and probability curvature, in that order. The historical runs used eight time nodes. The update counts below are the reported full CAL counts.

| Dataset | Upstream | Anchors | Regularization | CAL updates |
| --- | --- | --- | --- | ---: |
| Germany | Transformer-QR | 1, 8, 50, 90, 99 | 0.01, 0.01, 0 | 61,056 |
| Germany | iTransformer-QR | 1, 9, 50, 89, 99 | 0.01, 0.01, 0 | 52,864 |
| Germany | QR-GRU | 1, 10, 50, 90, 99 | 0, 0, 0.01 | 108,112 |
| Germany | QR-LSTM | 1, 40, 50, 92, 99 | 0.01, 0.01, 0.01 | 64,384 |
| Germany | QRF | 1, 13, 50, 90, 99 | 0.01, 0.01, 0 | 5,504 |
| Germany | XGBoost-QR | 1, 50, 99 | 0.01, 0.01, 0.01 | 2,816 |
| Germany | LightGBM-QR | 1, 50, 99 | 0.01, 0.01, 0.0001 | 7,168 |
| NIST | Transformer-QR | 1, 10, 50, 90, 99 | 0.01, 0.0001, 0.01 | 95,840 |
| NIST | iTransformer-QR | 1, 10, 50, 92, 99 | 0, 0.0001, 0 | 640 |
| NIST | QR-GRU | 1, 10, 50, 90, 99 | 0.01, 0, 0.0001 | 96,848 |
| NIST | QR-LSTM | 1, 10, 50, 90, 99 | 0.01, 0, 0 | 96,848 |
| NIST | QRF | 1, 10, 50, 90, 99 | 0.01, 0.01, 0.01 | 83,792 |
| NIST | XGBoost-QR | 1, 50, 99 | 0.01, 0.01, 0.01 | 6,144 |
| NIST | LightGBM-QR | 1, 10, 50, 90, 99 | 0.01, 0.01, 0.01 | 59,312 |

With a specified configuration, use `abqcr fit` on a full CAL export. This example uses the Germany XGBoost-QR settings with the portable CPU solver. The prediction and scoring commands also work with the model saved by `fit_cal`.

```bash
abqcr fit --input CAL.csv --model fitted.npz --anchors 1 50 99 --time-knots 8 --lead-bounds 720 2145 --regularization 0.01 0.01 0.01 --updates 2816
abqcr predict --input EVAL_forecasts.csv --model fitted.npz --output calibrated.csv
abqcr score --predictions calibrated.csv --observations EVAL_observations.csv
```

Use the [CSV schemas](README.md#fit-a-specified-layout-from-csv) for these local inputs. EVAL forecasts and observations must contain the same unique IDs. Score the frozen model's issued curves on the common daylight EVAL records. QS, CRPS and MAE are in per unit, and CE is in percentage points.
