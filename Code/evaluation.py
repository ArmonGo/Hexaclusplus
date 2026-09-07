import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from typing import Optional, Dict, Any, List, Tuple
from load_data import inverse_transform_label, inverse_transform_std, inverse_transform_interval_bounds


def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainties: np.ndarray,
                         n_bins: int = 10,
                         save_path: Optional[str] = None,
                         show_plot: bool = False) -> Dict[str, Any]:
    """
    Evaluate calibration of uncertainty estimates.
    """
    y_true        = np.asarray(y_true)
    y_pred        = np.asarray(y_pred)
    uncertainties = np.asarray(uncertainties)
    errors = np.abs(y_true - y_pred)
    sorted_idx          = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_idx]
    sorted_errors        = errors[sorted_idx]
    bin_size = len(sorted_uncertainties) // n_bins
    bin_means_uncertainty, bin_means_error, bin_stds_error = [], [], []
    bin_rmv, bin_rmse = [], []   # per-bin root-mean-variance and RMSE 

    for i in range(n_bins):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_uncertainties)
        seg_unc = sorted_uncertainties[start:end]
        seg_err = sorted_errors[start:end] # |y - mu| in this bin
        bin_means_uncertainty.append(np.mean(seg_unc))
        bin_means_error.append(np.mean(seg_err))
        bin_stds_error.append(np.std(seg_err))
        bin_rmv.append(np.sqrt(np.mean(seg_unc ** 2))) # RMV_j = sqrt(mean sigma^2)
        bin_rmse.append(np.sqrt(np.mean(seg_err ** 2)))  # RMSE_j = sqrt(mean (y-mu)^2)

    mad_factor = np.sqrt(2 / np.pi)   # ~0.7979
    bin_means_uncertainty = np.array(bin_means_uncertainty)
    bin_means_error = np.array(bin_means_error)
    expected_error = mad_factor * bin_means_uncertainty   # calibrated MAE per bin

    correlation, p_value = stats.pearsonr(bin_means_uncertainty, bin_means_error)
    ece  = np.mean(np.abs(expected_error - bin_means_error))
    sharpness = np.mean(uncertainties)   # mean predicted std in the given scale

    # ENCE 
    bin_rmv  = np.asarray(bin_rmv)
    bin_rmse = np.asarray(bin_rmse)
    valid    = bin_rmv > 0
    ence     = float(np.mean(np.abs(bin_rmv[valid] - bin_rmse[valid]) / bin_rmv[valid])) \
               if valid.any() else float('nan')

    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.errorbar(bin_means_uncertainty, bin_means_error, yerr=bin_stds_error,
                    fmt='o-', linewidth=2, markersize=8, capsize=5, label='Observed')
        max_val = max(max(bin_means_uncertainty), max(bin_means_error))
        # Perfect calibration: MAE = sqrt(2/pi) * sigma (slope < 1, not the diagonal)
        ax.plot([0, max_val], [0, mad_factor * max_val], 'r--', linewidth=2,
                label='Perfect calibration')
        ax.set_xlabel('Predicted Uncertainty (Std Dev)')
        ax.set_ylabel('Observed Error (MAE)')
        ax.set_title(f'Calibration Plot — ECE={ece:.4f}, r={correlation:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    return {
        'ece': ece,
        'ence': ence,
        'correlation': correlation,
        'p_value': p_value,
        'sharpness': sharpness,
        'bin_means_uncertainty': bin_means_uncertainty,
        'bin_means_error': bin_means_error,
        'bin_rmv': bin_rmv,
        'bin_rmse': bin_rmse,
    }


def evaluate_prediction_intervals(y_true: np.ndarray,
                                  lower_bounds: np.ndarray,
                                  upper_bounds: np.ndarray,
                                  confidence_level: float = 0.95,
                                  cwc_eta: float = 50.0) -> Dict[str, Any]:
    """
    Evaluate prediction interval coverage and width.
    """
    y_true = np.asarray(y_true)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    within = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    picp = float(np.mean(within))                 # coverage
    mpiw = float(np.mean(upper_bounds - lower_bounds))

    # CWC = NMPIW * (1 + gamma * exp(-eta * (PICP - mu))), gamma = 1 if PICP < mu else 0.
    y_range = float(np.max(y_true) - np.min(y_true))
    nmpiw   = mpiw / y_range if y_range > 0 else float('nan')
    gamma   = 1.0 if picp < confidence_level else 0.0
    cwc     = nmpiw * (1.0 + gamma * np.exp(-cwc_eta * (picp - confidence_level)))
    return {
        'actual_coverage':     picp,
        'expected_coverage':   confidence_level,
        'mean_interval_width': mpiw,
        'coverage_difference': picp - confidence_level,
        'nmpiw':               nmpiw,
        'cwc':                 float(cwc) }

def print_comparison_results(results: Dict[str, Dict[str, Any]],
                             label_scaler=None,
                             label_cols: Optional[List[str]] = None,
                             confidence_level: float = 0.95,
                             order: Optional[List[str]] = None,
                             rename: Optional[Dict[str, str]] = None) -> pd.DataFrame:

    has_scaler = (label_scaler is not None) and (label_cols is not None)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    rows = []

    for model_name, result in results.items():
        y_norm    = np.asarray(result['test_y'])
        pred_norm = np.asarray(result['test_pred'])
        std_norm  = result['test_std']
        std_norm  = np.asarray(std_norm) if std_norm is not None else None
        if has_scaler:
            y_eval = inverse_transform_label(y_norm,    label_scaler, label_cols)
            pred_eval = inverse_transform_label(pred_norm, label_scaler, label_cols)
            if std_norm is not None:
                std_eval   = inverse_transform_std(pred_norm, std_norm,label_scaler, label_cols)
                lower_eval, upper_eval = inverse_transform_interval_bounds(pred_norm, std_norm, label_scaler, label_cols, n_sigma=z_score)
            else:
                std_eval = lower_eval = upper_eval = None
        else:
            y_eval = y_norm
            pred_eval = pred_norm
            std_eval = std_norm
            if std_norm is not None:
                lower_eval = pred_norm - z_score * std_norm
                upper_eval = pred_norm + z_score * std_norm
            else:
                lower_eval = upper_eval = None
        rmse = float(np.sqrt(mean_squared_error(y_eval, pred_eval)))
        mae = float(mean_absolute_error(y_eval, pred_eval))
        r2 = float(r2_score(y_eval, pred_eval))

        if std_eval is not None:
            interval_metrics = evaluate_prediction_intervals(y_eval, lower_eval, upper_eval, confidence_level=confidence_level)
            calib = evaluate_calibration(y_eval, pred_eval, std_eval, n_bins=10)

            interval_width = interval_metrics['mean_interval_width']
            coverage = interval_metrics['actual_coverage']
            cwc = interval_metrics['cwc']
            ece = calib['ece']
            ence = calib['ence']
            correlation = calib['correlation']
            p_value = calib['p_value']
            sharpness = calib['sharpness']
        else:
            raise ValueError(f"Model '{model_name}' does not provide uncertainty estimates (std).")

        def fomat_row(v, decimals=4):
            return f"{v:.{decimals}f}" if v is not None else "N/A"
        scale_note = "(orig.)" if has_scaler else "(norm.)"
        tuning_time = result.get('tuning_time_s')
        n_configs = result.get('n_configs')
        time_per = result.get('time_per_config_s')
        rows.append({
            'Model': model_name,
            f'RMSE {scale_note}': fomat_row(rmse),
            f'MAE {scale_note}':  fomat_row(mae),
           # 'R²': fomat_row(r2),
            f'Sharpness {scale_note}': fomat_row(sharpness),
           # f'95% PI Width {scale_note}': fomat_row(interval_width),
            f'Coverage ({int(confidence_level*100)}%)': fomat_row(coverage),
            'CWC': fomat_row(cwc),
            f'ECE {scale_note}': fomat_row(ece),
            'ENCE': fomat_row(ence),
            'Calib. Corr': fomat_row(correlation),
           # 'Calib. p-value': fomat_row(p_value, decimals=4),
            'Tuning Time (s)': fomat_row(tuning_time, decimals=1),
            'N Configs': str(n_configs) if n_configs is not None else 'N/A',
            'Time/Config (s)': fomat_row(time_per, decimals=1)})
    df = pd.DataFrame(rows).set_index('Model')
    if order is not None:
        present = [m for m in order if m in df.index]
        rest    = [m for m in df.index if m not in present]  # reorder
        df = df.reindex(present + rest)
    if rename is not None:
        df = df.rename(index=rename)
    return df

