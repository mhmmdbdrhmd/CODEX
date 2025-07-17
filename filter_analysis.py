
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate

# Fixed ranges for the Extrema MAE heatmaps
HEATMAP_REF_RANGE = np.linspace(60.0, 130.0, 100)
HEATMAP_SCALE_RANGE = np.linspace(0.5, 1.5, 100)

# -----------------------------------------------------------------------------------
# 1. Load & Preprocess Function
# -----------------------------------------------------------------------------------
def load_and_preprocess(path):
    """Load CSV and preprocess sensor, target, and speed columns.

    The function automatically detects whether redundant (RC) sensors were used
    by checking the ``*SE2`` and ``*Be_SE2`` columns. Depending on that, the raw
    left/right sums are computed with either 2 or 4 sensors per side. The raw
    hitch angle proxy is then derived from these sums. The function also
    interpolates zero dropouts in the reference angle, smooths speed, and
    applies a causal Hampel filter to the raw signal.

    Returns
    -------
    df : DataFrame
        Preprocessed data with added columns ``raw`` and ``cleaned_raw``.
    rc_used : bool
        Whether redundant sensors were detected in this file.
    """

    df = pd.read_csv(path, sep=';')

    # Identify relevant columns by regex
    angle_col = df.filter(regex='Deichsel').columns[0]
    speed_col = df.filter(regex='Geschwindigkeit').columns[0]
    Lse = df.filter(regex='Durchschnitt_L_SE').columns[0]
    Lbe = df.filter(regex='Durchschnitt_L_Be_SE').columns[0]
    Rse = df.filter(regex='Durchschnitt_R_SE').columns[0]
    Rbe = df.filter(regex='Durchschnitt_R_[Bb]e_SE').columns[0]

    # Determine whether redundant sensors were used (any non-zero in *SE2 columns)
    Lse2 = df.filter(regex='Durchschnitt_L_SE2').columns[0]
    Rse2 = df.filter(regex='Durchschnitt_R_SE2').columns[0]
    Lbe2 = df.filter(regex='Durchschnitt_L_Be_SE2').columns[0]
    Rbe2 = df.filter(regex='Durchschnitt_R_[Bb]e_SE2').columns[0]
    rc_used = not df[[Lse2, Rse2, Lbe2, Rbe2]].eq(0).to_numpy().all()

    if rc_used:
        df['L'] = (df[Lse] + df[Lse2] + df[Lbe] + df[Lbe2]) + 32.0
        df['R'] = (df[Rse] + df[Rse2] + df[Rbe] + df[Rbe2]) + 32.0
    else:
        df['L'] = df[Lse] + df[Lbe] + 16.0
        df['R'] = df[Rse] + df[Rbe] + 16.0

    # Raw hitch-angle proxy
    df['raw'] = 90 + (((df['L'] + df['R']) / 2) * (df['L'] - df['R'])) / (
        df['L'] * df['R']) * 100

    # Clean target: replace zeros with NaN, then interpolate
    df['target_clean'] = df[angle_col].replace(0, np.nan).interpolate(method='linear', limit_direction='both')

    # Speed preprocessing:
    speed = df[speed_col].clip(lower=0)
    non_zero = np.where(speed != 0)[0]
    if len(non_zero) > 0:
        fnz, lnz = non_zero[0], non_zero[-1]
        speed.iloc[fnz:lnz+1] = speed.iloc[fnz:lnz+1].replace(0, np.nan)
    if not isinstance(speed, pd.Series):
        speed = pd.Series(speed)
    speed = speed.interpolate().rolling(window=5, center=True, min_periods=1).mean()
    if isinstance(speed, pd.Series):
        speed = speed.fillna(0)
    df['speed'] = speed

    # Hampel filter on 'raw' to remove spikes (causal window=5)
    x = df['raw'].copy()
    k, w = 1.4826, 5
    for i in range(w, len(x)-w):
        window = x[i-w:i+w+1]
        med = np.median(window)
        mad = k * np.median(np.abs(window - med))
        if abs(x[i] - med) > 3 * mad:
            x[i] = med
    df['cleaned_raw'] = x

    return df, rc_used

# -----------------------------------------------------------------------------------
# 2. Extrema Detection & Alignment Functions
# -----------------------------------------------------------------------------------
def detect_extrema(arr):
    """
    Detects local maxima (peaks) and minima (valleys) in arr using 10% prominence.
    Returns arrays of peak indices and valley indices.
    """
    prom = 0.1 * (np.nanmax(arr) - np.nanmin(arr))
    peaks, _ = find_peaks(arr, prominence=prom)
    valleys, _ = find_peaks(-arr, prominence=prom)
    return peaks, valleys

def align_by_extrema(y, t):
    """
    Aligns series y to reference series t using extrema matching:
    1. Detect true peaks/valleys (tp, tv) in t.
    2. Detect peaks/valleys in y.
    3. Compute shifts between each true extremum and its nearest filtered extremum.
    4. Median of shifts = lag. Roll y by -lag to align.
    Returns rolled y, true peaks, true valleys, and lag.
    """
    tp, tv = detect_extrema(t)
    prom_y = 0.1 * (np.nanmax(y) - np.nanmin(y))
    yp, _ = find_peaks(y, prominence=prom_y)
    yv, _ = find_peaks(-y, prominence=prom_y)
    shifts = []
    for idx in np.concatenate([tp, tv]):
        if len(yp) > 0:
            shifts.append(yp[np.argmin(np.abs(yp - idx))] - idx)
        if len(yv) > 0:
            shifts.append(yv[np.argmin(np.abs(yv - idx))] - idx)
    lag = int(np.median(shifts)) if shifts else 0

    # Shift by lag without wrap-around, padding with edge values
    def shift_with_pad(arr, shift):
        if shift > 0:
            shifted = np.empty_like(arr)
            shifted[:-shift] = arr[shift:]
            shifted[-shift:] = arr[-1]
        elif shift < 0:
            shift = -shift
            shifted = np.empty_like(arr)
            shifted[shift:] = arr[:-shift]
            shifted[:shift] = arr[0]
        else:
            shifted = arr.copy()
        return shifted

    y_shifted = shift_with_pad(y, lag)
    return y_shifted, tp, tv, lag

# ------------------------------------------------------------------------------
# Helper: Grid Search Scaling to Minimize Extrema MAE
# ------------------------------------------------------------------------------
def optimize_scaling(t, y_al, ext_idx, ref_range=None, k_range=None):
    """Grid search to optimize reference angle and scale factor using MAE only.

    Parameters
    ----------
    t : ndarray
        True signal (reference).
    y_al : ndarray
        Filtered signal already aligned to t.
    ext_idx : array-like
        Indices of extrema in the true signal.
    ref_range : iterable, optional
        Reference angles to test (default 80..100).
    k_range : iterable, optional
        Scale factors to test (default 0.9..1.1).

    Returns
    -------
    best_scaled : ndarray
        Scaled version of ``y_al`` giving the lowest MAE.
    best_ref : float
        Reference angle used for the best scaling.
    best_k : float
        Scale factor used for the best scaling.
    best_mae : float
        The resulting MAE.
    """
    if ref_range is None:
        # Determine range of aligned signal and focus search around its mean
        lo, hi = np.nanmin(y_al), np.nanmax(y_al)
        mid = (lo + hi) / 2.0
        width = (hi - lo) / 2.0
        r = np.linspace(-1.0, 1.0, 200)
        ref_range = mid + width * np.sign(r) * np.abs(r) ** 1.5
    if k_range is None:
        # Expanded scale search (0.5..1.5) with finer resolution
        k_range = np.linspace(0.5, 1.5, 200)

    mask = ~np.isnan(t) & ~np.isnan(y_al)

    best_mae = np.inf
    best_scaled = y_al.copy()
    best_ref = 90.0
    best_k = 1.0

    lo, hi = np.nanmin(y_al), np.nanmax(y_al)
    for ref in ref_range:
        ref_c = np.clip(ref, lo, hi)
        for k in k_range:
            scaled = k * (y_al - ref_c) + ref_c
            mae = mean_absolute_error(t[mask], scaled[mask])
            if mae < best_mae:
                best_mae = mae
                best_scaled = scaled
                best_ref = ref_c
                best_k = k

    return best_scaled, best_ref, best_k, best_mae

# ------------------------------------------------------------------------------
# Helper: Compute grid of Extrema MAE for heatmap
# ------------------------------------------------------------------------------
def extrema_mae_grid(t, y_al, ext_idx, ref_range, k_range):
    """Return matrix of Extrema MAE for each (ref, scale) pair."""
    grid = np.zeros((len(ref_range), len(k_range)))
    lo, hi = np.nanmin(y_al), np.nanmax(y_al)
    for i, ref in enumerate(ref_range):
        ref_c = np.clip(ref, lo, hi)
        for j, k in enumerate(k_range):
            scaled = k * (y_al - ref_c) + ref_c
            grid[i, j] = mean_absolute_error(t[ext_idx], scaled[ext_idx])

    return grid, ref_range, k_range

# -----------------------------------------------------------------------------------
# 3. KF_inv (Speed‐Aware Kalman Filter)
# -----------------------------------------------------------------------------------
def kf_inv(x, v):
    """
    One-state Kalman filter using speed v to adapt process noise:
    Q = 1e-2 + 1e-1 * (1 / max(v[i], 0.1)), R = 1.0.
    Returns filtered array y.
    """
    y = np.zeros_like(x)
    P = np.zeros_like(x)
    y[0], P[0] = x[0], 1.0
    for i in range(1, len(x)):
        vs = max(v[i], 0.1)
        Q = 1e-2 + 1e-1 * (1 / vs)
        Pp = P[i-1] + Q
        K = Pp / (Pp + 1.0)
        y[i] = y[i-1] + K * (x[i] - y[i-1])
        P[i] = (1 - K) * Pp
    return y

# -----------------------------------------------------------------------------------
# 4. Main Processing Loop for All Recordings
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # Directory containing all CSV recordings
    recordings_dir = "recordings"

    # Optional: Exclude first N seconds globally; set None to disable
    exclude_first_seconds = None  # in seconds

    # Per-file trimming in seconds [start_trim, end_trim]
    trim_seconds = {
        "log_1622_64296": [0, 500],
        "log_1618_76251": [0, 100],
        "log_1626_93118": [100, 0],
    }

    # Filters to show in the generated plots. All filters are always computed
    # for the performance table regardless of this list.
    plot_filters = ["KF_inv"]

    # Initialize a list to collect metrics for all files
    all_metrics = []

    # Loop through each CSV file in the recordings directory
    for fname in os.listdir(recordings_dir):
        if not fname.lower().endswith(".csv"):
            continue

        file_path = os.path.join(recordings_dir, fname)
        print(f"\nProcessing recording: {fname}")

        base = fname[:-4]

        # 4.1 Load and preprocess with automatic RC detection
        df, rc_used = load_and_preprocess(file_path)
        print(f"  Redundant sensors used: {rc_used}")

        # Trim beginning/end based on per-file settings
        if base in trim_seconds:
            start_trim, end_trim = trim_seconds[base]
            if start_trim > 0:
                df = df.iloc[start_trim:].reset_index(drop=True)
            if end_trim > 0:
                df = df.iloc[:-end_trim] if end_trim < len(df) else df.iloc[0:0]
                df = df.reset_index(drop=True)

        # 4.2 Optionally exclude initial period based on timestamp column (ms)
        if exclude_first_seconds is not None:
            t0 = df["ESP Time"].iloc[0]
            cutoff = t0 + int(exclude_first_seconds * 1000)
            df = df[df["ESP Time"] >= cutoff].reset_index(drop=True)

        # 4.3 Extract arrays for processing
        idx = df.index
        t_clean = df["target_clean"]
        speed = df["speed"]
        raw = df["cleaned_raw"]
        if isinstance(idx, pd.Index):
            idx = idx.to_numpy()
        if isinstance(t_clean, pd.Series):
            t_clean = t_clean.to_numpy()
        if isinstance(speed, pd.Series):
            speed = speed.to_numpy()
        if isinstance(raw, pd.Series):
            raw = raw.to_numpy()

        # 4.4 Apply filters
        kf_out = kf_inv(raw, speed)
        wl = 101 if len(raw) >= 101 else (len(raw)//2)*2 + 1
        sg_out = savgol_filter(raw, wl, 3)
        kf_on_sg_out = kf_inv(sg_out, speed)

        # 4.5 Compute metrics and scaling for each filter on this file
        params = {}
        aligned = {}
        scaled = {}
        ext_maes = {}
        tp_all = tv_all = None

        for name, y in [("KF_inv", kf_out), ("SG", sg_out), ("KF_on_SG", kf_on_sg_out)]:
            y_al, tp, tv, lag = align_by_extrema(y, t_clean)
            if tp_all is None:
                tp_all, tv_all = tp, tv

            mask = ~np.isnan(t_clean) & ~np.isnan(y_al)
            rmse = np.sqrt(mean_squared_error(t_clean[mask], y_al[mask]))
            mae = mean_absolute_error(t_clean[mask], y_al[mask])
            mape_pk = np.mean(np.abs(y_al[tp] - t_clean[tp])) if len(tp) > 0 else np.nan
            mave_vl = np.mean(np.abs(y_al[tv] - t_clean[tv])) if len(tv) > 0 else np.nan
            ext_idx = np.concatenate([tp, tv])
            ext_mae = np.mean(np.abs(y_al[ext_idx] - t_clean[ext_idx])) if len(ext_idx) > 0 else np.nan

            y_al_scaled, ref_opt, k_opt, _ = optimize_scaling(t_clean, y_al, ext_idx)
            y_scaled = k_opt * (y - ref_opt) + ref_opt

            rmse_scaled = np.sqrt(mean_squared_error(t_clean[mask], y_al_scaled[mask]))
            mae_scaled = mean_absolute_error(t_clean[mask], y_al_scaled[mask])
            ext_mae_scaled = (np.mean(np.abs(y_al_scaled[ext_idx] - t_clean[ext_idx]))
                              if len(ext_idx) > 0 else np.nan)

            all_metrics.append((fname[:-4], name, rmse, mae, ext_mae, ext_mae_scaled,
                                mape_pk, mave_vl, lag, ref_opt, k_opt,
                                rmse_scaled, mae_scaled))

            ext_maes[name] = ext_mae

            print(f"  {name}: MAE={mae:.3f}, ExtMAE={ext_mae:.3f}, "
                  f"MAE_scaled={mae_scaled:.3f}, ExtMAE_scaled={ext_mae_scaled:.3f}")

            params[name] = (ref_opt, k_opt)
            aligned[name] = (y_al, lag, y_al_scaled)
            scaled[name] = y_scaled

        # 4.6 Plot General + Alignment for this file
        tp, tv = tp_all, tv_all

        color_map = {"KF_inv": "k", "SG": "r", "KF_on_SG": "m"}
        label_map = {"KF_inv": "KF_inv", "SG": "SG Filter", "KF_on_SG": "KF_on_SG"}
        lw_map = {"KF_inv": 1.2, "SG": 1, "KF_on_SG": 1}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top subplot: unaligned signals + speed
        ax1.plot(idx, t_clean, 'b-', lw=1, label='True')
        outputs = {"KF_inv": kf_out, "SG": sg_out, "KF_on_SG": kf_on_sg_out}
        for name in plot_filters:
            ax1.plot(idx, outputs[name], color_map[name], lw=lw_map[name],
                     label=label_map[name])
            ax1.plot(idx, scaled[name], color_map[name], linestyle=':', lw=1,
                     label=f'{name}_scaled')
        ax1.set_ylabel('Angle')
        ax1.set_title(f"General + Alignment: {fname[:-4]}")
        ax1.legend(loc='upper left')
        ax1_r = ax1.twinx()
        ax1_r.plot(idx, speed, 'g-', lw=1, label='Speed')
        ax1_r.set_ylabel('Speed')
        ax1_r.legend(loc='upper right')

        # Bottom subplot: aligned signals with true extrema markers + speed
        ax2.plot(idx, t_clean, 'b-', lw=1, label='True')
        for name in plot_filters:
            y_al, lag, y_al_scaled = aligned[name]
            ax2.plot(idx, y_al, color_map[name], linestyle='--', lw=lw_map[name],
                     label=f'{name}_aligned (lag={lag})')
            ax2.plot(idx, y_al_scaled, color_map[name], linestyle=':', lw=1,
                     label=f'{name}_scaled')
        if tp is not None and len(tp) > 0:
            ax2.scatter(idx[tp], t_clean[tp],   c='cyan',   marker='o', label='True Peaks')
        if tv is not None and len(tv) > 0:
            ax2.scatter(idx[tv], t_clean[tv],   c='magenta', marker='x', label='True Valleys')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Angle')
        ax2.legend(loc='upper left')
        ax2_r = ax2.twinx()
        ax2_r.plot(idx, speed, 'g-', lw=1)
        ax2_r.set_ylabel('Speed')

        plt.tight_layout()
        fig.savefig("results/General_"+fname[:-4]+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4.7 Plot Detail View (2×3) using same y-limits
        left_ylim = ax1.get_ylim()
        right_ylim = ax1_r.get_ylim()
        n = len(df)
        periods = [(i*(n//6), (i+1)*(n//6) if i<5 else n) for i in range(6)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, axd in enumerate(axes.flatten()):
            s, e = periods[i]
            axd.plot(idx[s:e], t_clean[s:e], 'b-', lw=1)
            for name in plot_filters:
                axd.plot(idx[s:e], outputs[name][s:e], color_map[name], lw=lw_map[name])
                axd.plot(idx[s:e], scaled[name][s:e], color_map[name], linestyle=':', lw=1)
            axd.set_title(f"Segment {i+1}")
            axd.set_ylim(left_ylim)
            axd_r = axd.twinx()
            axd_r.plot(idx[s:e], speed[s:e], 'g-', lw=1)
            axd_r.set_ylim(right_ylim)

        fig.suptitle(f"Detail View (2×3): {fname[:-4]}", y=1.02)
        legend_lines = [Line2D([0],[0],color='b')]
        legend_labels = ['True']
        for name in plot_filters:
            legend_lines.append(Line2D([0],[0],color=color_map[name]))
            legend_labels.append(label_map[name])
            legend_lines.append(Line2D([0],[0],color=color_map[name], linestyle=':'))
            legend_labels.append(f'{name}_scaled')
        legend_lines.append(Line2D([0],[0],color='g'))
        legend_labels.append('Speed')
        fig.legend(legend_lines, legend_labels, loc='upper right')
        plt.tight_layout()
        fig.savefig("results/Detail_"+fname[:-4]+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4.8 Plot heatmap of Extrema MAE over scaling parameters
        heat_filter = plot_filters[0] if plot_filters else "KF_inv"
        y_al_heat, _, _ = aligned[heat_filter]
        baseline = ext_maes.get(heat_filter, np.nan)
        if tp_all is not None and tv_all is not None:
            ext_idx = np.concatenate([tp_all, tv_all])
        else:
            ext_idx = np.array([], dtype=int)
        grid_raw, rr, kk = extrema_mae_grid(
            t_clean,
            y_al_heat,
            ext_idx,
            HEATMAP_REF_RANGE,
            HEATMAP_SCALE_RANGE,
        )
        grid_norm = grid_raw / baseline if baseline else grid_raw

        mask_raw = np.ma.array(grid_raw, mask=grid_raw > baseline)
        mask_norm = np.ma.array(grid_norm, mask=grid_norm > 1.0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = [
            "Raw Extrema MAE",
            "Normalized Extrema MAE",
            "Raw MAE (masked)",
            "Normalized MAE (masked)",
        ]
        data_mats = [grid_raw, grid_norm, mask_raw, mask_norm]
        for ax, data, title in zip(axes.flat, data_mats, titles):
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=[kk[0], kk[-1], rr[0], rr[-1]],
                cmap="viridis",
            )
            if isinstance(data, np.ma.MaskedArray):
                im.cmap.set_bad("black")
            ax.set_xlabel("Scale k")
            ax.set_ylabel("Reference angle")
            ax.set_title(title)
            fig.colorbar(im, ax=ax).set_label(title)
        fig.suptitle(f"Extrema MAE Heatmaps (KF_inv): {fname[:-4]}")
        plt.tight_layout()
        fig.savefig("results/Heatmap_"+fname[:-4]+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)


    # 4.8 After processing all files, display combined metrics
    df_all_metrics = pd.DataFrame(all_metrics, columns=[
        "Filename", "Method", "RMSE", "MAE", "Extrema_MAE",
        "Extrema_MAE_scaled", "MAPE_pk", "MAVE_vl", "Lag",
        "Ref_Angle", "Scale_k", "RMSE_scaled", "MAE_scaled"
    ])  # type: ignore
    print("\n=== Combined Performance Metrics for All Recordings ===")
    print(df_all_metrics.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    df_all_metrics.to_csv("results/performance.csv", index=False)

    # Update README results section
    try:
        table_start = "<!-- RESULTS_TABLE_START -->"
        table_end = "<!-- RESULTS_TABLE_END -->"
        plots_start = "<!-- RESULTS_PLOTS_START -->"
        plots_end = "<!-- RESULTS_PLOTS_END -->"

        with open("README.md", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Update performance table
        if table_start in lines and table_end in lines:
            s = lines.index(table_start)
            e = lines.index(table_end)
            table_md = tabulate(df_all_metrics, headers="keys", tablefmt="pipe", showindex=False)
            table_block = ["<details><summary>Performance Summary</summary>", ""] + table_md.splitlines() + ["</details>"]
            lines = lines[:s+1] + table_block + lines[e:]

        # Update plot subsections
        if plots_start in lines and plots_end in lines:
            s = lines.index(plots_start)
            e = lines.index(plots_end)
            file_sections = []
            for fname in sorted(df_all_metrics['Filename'].unique()):
                file_sections.extend([
                    f"<details><summary>{fname}</summary>",
                    "",
                    f"![General {fname}](results/General_{fname}.png)",
                    f"![Detail {fname}](results/Detail_{fname}.png)",
                    f"![Heatmap {fname}](results/Heatmap_{fname}.png)",
                    "",
                    "</details>"
                ])
            lines = lines[:s+1] + file_sections + lines[e:]

        with open("README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        print(f"Failed to update README results: {exc}")

