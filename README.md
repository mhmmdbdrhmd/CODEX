
# Angle Sensor Filtering & Extremum Tracking

## Overview
Raw light-sensor arrays provide a noisy glimpse of how the tractor and trailer are oriented. The goal is to turn those inexpensive readings, together with vehicle speed, into a stable hitch-angle signal. A high-quality sensor supplies the ground truth used for calibration and evaluation.
Rather than hard‐coding specific filenames, this workflow is designed to operate on **any new recording** (CSV file) you provide. As long as each CSV follows the same column conventions (left/right sensor sums, a reference angle column, a speed column, and a timestamp), the same preprocessing, filtering, alignment, and evaluation steps will apply.

## Main Task

Our goal is to provide a clear example pipeline that:
1. Reads any CSV from `recordings/`.
2. Derives a hitch-angle estimate from the light-sensor arrays.
3. Cleans the reference sensor and vehicle speed columns.
4. Runs a speed-aware Kalman filter and optional alternatives.
5. Aligns the filtered result with the reference and logs error metrics.
6. Saves reproducible plots in `results/` for inspection.

Think of this repository as a starting point for developers to experiment with filtering methods and contribute improvements.


---

## Directory Structure

```
.
├── README.md                  ← This file (generic instructions)
├── filter_analysis.py         ← Python script (load any CSV, process, plot, metrics)
├── recordings/                ← Place one or more CSV recordings here
│   ├── log_1618_76251.csv     ← Example recording file
│   ├── log_1622_64296.csv
│   └── log_1626_93118.csv
└── results/                   ← Auto-generated figures and `performance.csv`
```

- **filter_analysis.py**: Contains all code for loading, preprocessing, filtering, alignment, and performance evaluation. It automatically processes **every CSV recording** in the `recordings/` folder.
- **recordings/**: Directory where you place each new CSV log. Each file must contain the following columns using the exact German names from the logger:
  - `Durchschnitt_L_SE` and `Durchschnitt_L_Be_SE` – left light-sensor sums,
  - `Durchschnitt_R_SE` and `Durchschnitt_R_Be_SE` – right light-sensor sums,
  - `Deichsel_Angle` – drawbar (hitch) angle reference,
  - `Geschwindigkeit` – vehicle speed,
  - `ESP Time` – timestamp in milliseconds.
- **results/**: Created automatically when you run the script. Holds PNG graphs and a `performance.csv` summary table.

---

## 1. What We’re Trying to Accomplish

1. **Compute a raw hitch‐angle proxy** from left/right sensor sums.
2. **Clean the target reference** by interpolating any zero‐dropouts.
3. **Preprocess speed** by clipping negative values, interpolating zeros (only between first and last movement), and applying a small rolling‐average to remove jitter.
4. **Remove spikes** from the raw proxy with a causal Hampel filter (window=5).
5. **Apply a speed‐aware Kalman filter** (KF_inv) in real time, where process‐noise is scaled by 1/speed.
6. **Detect local extrema (peaks & valleys)** in the clean reference signal to quantify how well each filter preserves high/low points.
7. **Align** each filtered output to the reference by matching extremum indices, so that peak/valley errors can be measured at the correct time points.
8. **Compute metrics** (RMSE, MAE, plus specialized extrema‐MAE) to evaluate performance.
9. **Optionally test alternative filters** (e.g. Savitzky–Golay and hybrid KF_on_SG) on the same recording and compare results.

Because each recording file uses identical column names/patterns, you can simply drop new CSVs into the `recordings/` directory and re-run the script.

---

## 2. Preprocessing Steps (Single Recording)

When you run the analysis script on any new recording:

1. **Load the CSV** into a DataFrame.
2. **Identify columns** via their column‐name patterns:
   - `Deichsel_Angle` (drawbar angle reference).
   - `Geschwindigkeit` (speed).
   - `Durchschnitt_L_SE` and `Durchschnitt_L_Be_SE` (left light-sensor sums).
   - `Durchschnitt_R_SE` and `Durchschnitt_R_Be_SE` (right light-sensor sums).
3. **Compute the raw proxy**:
  - Define
    ```
      offset = 16  # per-file light-sensor offset
      L = (left‐sensor1 + left‐sensor2 + offset),
      R = (right‐sensor1 + right‐sensor2 + offset).
    ```
   - Compute  
     ```
       raw = 90 + (((L + R) / 2) * (L - R)) / (L * R) * 100.
     ```
4. **Clean the target**:
   - Replace zeros with NaN, then interpolate linearly (forward and backward).
5. **Preprocess speed**:
   - Clip negative values to zero.
   - Between the first and last nonzero sample, replace zeros with NaN and interpolate.
   - Apply a 5‐sample centered rolling mean, filling any NaNs at the ends with zero.
   - This yields a smooth, causal speed estimate for use in the Kalman filter.
6. **Suppress spikes** in the raw proxy using a **causal Hampel filter** (window=5).

---

## 3. Filtering Methods

Once preprocessing is complete, the script applies one or more of the following filters to `cleaned_raw`:

1. **KF_inv** (Speed‐Aware Kalman Filter)
2. **Savitzky–Golay (SG)**
3. **KF_on_SG** (KF_inv applied to SG output)

Customize by commenting/uncommenting the desired method calls in `filter_analysis.py`.

---

## 4. Extrema Detection & Alignment (Single Recording)

1. **Detect True Extrema**: local maxima/minima of `target_clean` with 10% prominence.
2. **Detect Filtered Extrema**: local maxima/minima of each filtered output with 10% of that series’ range.
3. **Align by Extrema**:
   - For each true extremum index, find nearest filtered extremum index, compute shift = (filtered_idx - true_idx).
   - Take median(shift) = `lag`. Shift the filtered output by `-lag` without wrap-around, repeating the edge values.
   - Now filtered peaks/valleys coincide with true peaks/valleys.

---

## 5. Scaling Optimization

After alignment, each filtered output is rescaled by searching a dense grid of
reference angles and scale factors. Reference candidates are drawn from the
range of the aligned signal itself, with 200 points concentrated around the
signal mean and clamped to its min/max. For every reference, scale factors from
0.8 to 1.2 (100 steps) are tested. The pair that yields the lowest MAE is selected.
The resulting `ref_angle` and `scale_k` are logged for each filter.

---

## 6. Performance Metrics (Single Recording)

After alignment:
- **RMSE**: sqrt(mean squared error over all valid samples).
- **MAE**: mean absolute error over all valid samples.
- **MAPE_pk**: mean absolute error at true-peak indices.
- **MAVE_vl**: mean absolute error at true-valley indices.
- **Extrema_MAE**: mean absolute error over all true-peak+valley indices.
- **RMSE_scaled**: RMSE after scaling optimization.
- **MAE_scaled**: MAE after scaling optimization.
- **Extrema_MAE_scaled**: mean absolute error at true extrema after scaling.

---

## 7. How to Use

1. **Place recording(s)** (CSV files) in `recordings/`.
2. **Edit `filter_analysis.py`** if needed:
   - Adjust `exclude_first_seconds` (default = `None`).
   - Update the `trim_seconds` dictionary for per-file start/end trimming.
   - Optionally define per-file light-sensor offsets via the `sensor_offset`
     dictionary (default = `16`).
3. **Run**:
   ```
   python filter_analysis.py
   ```
   - The console prints performance metrics.
   - Two figures appear:
     1. **General + Alignment** (two‐subplot).
     2. **Detail View** (2×3).

4. **Add new recordings**:
   - Just drop additional CSVs into `recordings/` and run the script again.
5. **Generated figures** are saved in `results/` as PNG files when you run the
   script. These images are reproducible and don't need to be committed to
   version control.

---

## 8. Next Steps & Customization

- Adjust prominence thresholds or filter parameters as needed.
- Explore additional causal smoothing methods.
- Automate batch processing for multiple recordings.

---
## 9. Automated Runs

A GitHub Actions workflow (`run-analysis.yml`) installs Python **3.12**, installs the requirements, and then runs `python filter_analysis.py` whenever you push changes. The generated plots in `results/` are uploaded as workflow artifacts so you can inspect them without committing the PNG files.

---

## 10. Results Directory

Running the script creates a `results/` folder containing:

- `performance.csv` – table of metrics for every recording and method.
- `General_<recording>.png` – overview plot with alignment.
- `Detail_<recording>.png` – zoomed detail view.

`performance.csv` columns:

| Column | Meaning |
|-------|---------|
| `Filename` | source CSV log |
| `Method` | filtering method used |
| `RMSE`, `MAE` | overall error metrics |
| `Extrema_MAE` | error at detected peaks/valleys |
| `Extrema_MAE_scaled` | peak/valley error after scaling |
| `MAPE_pk` | mean absolute peak error |
| `MAVE_vl` | mean absolute valley error |
| `Lag` | alignment shift in samples |
| `Ref_Angle` | optimal reference angle |
| `Scale_k` | scale factor applied |
| `RMSE_scaled`, `MAE_scaled` | errors after scaling |

---

## 11. Results

The table and figures below are automatically refreshed from the `results/` folder whenever the CI workflow runs `filter_analysis.py`. They always show the most up-to-date results from the latest analysis.

<!-- RESULTS_TABLE_START -->
| Filename       | Method   |    RMSE |     MAE |   Extrema_MAE |   Extrema_MAE_scaled |   MAPE_pk |   MAVE_vl |   Lag |   Ref_Angle |   Scale_k |   RMSE_scaled |   MAE_scaled |
|:---------------|:---------|--------:|--------:|--------------:|---------------------:|----------:|----------:|------:|------------:|----------:|--------------:|-------------:|
| log_1626_93118 | KF_inv   | 3.50013 | 2.72786 |       3.24046 |              1.82055 |   4.81845 |   1.78385 |    14 |     86.7324 |  1.29397  |       1.63688 |      1.1243  |
| log_1626_93118 | SG       | 3.61859 | 2.75185 |       3.43375 |              1.75933 |   5.05427 |   1.93788 |     2 |     86.8341 |  1.29397  |       1.90762 |      1.27844 |
| log_1626_93118 | KF_on_SG | 3.6728  | 2.76034 |       3.50694 |              1.67498 |   5.22371 |   1.92222 |     6 |     86.9135 |  1.29397  |       2.00796 |      1.34863 |
| log_1618_76251 | KF_inv   | 2.47858 | 1.73186 |       3.42423 |              1.35454 |   2.27022 |   3.71274 |    19 |     94.418  |  0.876884 |       2.16629 |      1.36809 |
| log_1618_76251 | SG       | 2.59131 | 1.90651 |       2.00123 |              1.57672 |   2.94572 |   1.7651  |     4 |     93.3878 |  0.886935 |       2.30534 |      1.48839 |
| log_1618_76251 | KF_on_SG | 2.40439 | 1.80014 |       1.85107 |              1.89041 |   3.20227 |   1.51327 |    11 |     93.1637 |  0.88191  |       2.15722 |      1.39526 |
| log_1622_64296 | KF_inv   | 3.13407 | 2.51597 |       2.41009 |              1.92347 |   2.38969 |   2.43168 |    10 |    106.716  |  0.91206  |       2.81522 |      1.99782 |
| log_1622_64296 | SG       | 3.13703 | 2.46259 |       3.91841 |              3.6739  |   4.03643 |   3.79346 |    10 |    123.011  |  0.957286 |       2.8626  |      1.98251 |
| log_1622_64296 | KF_on_SG | 3.09645 | 2.47268 |       4.08145 |              3.99556 |   4.63121 |   3.49934 |    12 |    127.293  |  0.962312 |       2.86606 |      2.00294 |
<!-- RESULTS_TABLE_END -->

![General 1618](results/General_log_1618_76251.png)
![Detail 1618](results/Detail_log_1618_76251.png)
![General 1622](results/General_log_1622_64296.png)
![Detail 1622](results/Detail_log_1622_64296.png)
![General 1626](results/General_log_1626_93118.png)
![Detail 1626](results/Detail_log_1626_93118.png)

