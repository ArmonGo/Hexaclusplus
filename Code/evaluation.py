
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from typing import Optional, Dict, Any, List, Tuple

from load_data import (
    inverse_transform_label,
    inverse_transform_std,
    inverse_transform_interval_bounds,
)

def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainties: np.ndarray,
                         n_bins: int = 10,
                         save_path: Optional[str] = None,
                         show_plot: bool = False) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainties = np.asarray(uncertainties)

    errors = np.abs(y_true - y_pred)

    sorted_idx = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_idx]
    sorted_errors = errors[sorted_idx]

    bin_size = len(sorted_uncertainties) // n_bins
    bin_means_uncertainty, bin_means_error, bin_stds_error = [], [], []

    for i in range(n_bins):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_uncertainties)
        bin_means_uncertainty.append(np.mean(sorted_uncertainties[start:end]))
        bin_means_error.append(np.mean(sorted_errors[start:end]))
        bin_stds_error.append(np.std(sorted_errors[start:end]))

    mad_factor = np.sqrt(2 / np.pi) 
    bin_means_uncertainty = np.array(bin_means_uncertainty)
    bin_means_error = np.array(bin_means_error)
    expected_error = mad_factor * bin_means_uncertainty  

    correlation, p_value = stats.pearsonr(bin_means_uncertainty, bin_means_error)
    ece = np.mean(np.abs(expected_error - bin_means_error))
    sharpness = np.mean(uncertainties)   # mean predicted std in the given scale

    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.errorbar(bin_means_uncertainty, bin_means_error, yerr=bin_stds_error,
                    fmt='o-', linewidth=2, markersize=8, capsize=5, label='Observed')
        max_val = max(max(bin_means_uncertainty), max(bin_means_error))
       
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
        'correlation': correlation,
        'p_value': p_value,
        'sharpness': sharpness,
        'bin_means_uncertainty': bin_means_uncertainty,
        'bin_means_error': bin_means_error,
    }

def evaluate_prediction_intervals(y_true: np.ndarray,
                                  lower_bounds: np.ndarray,
                                  upper_bounds: np.ndarray,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
    
    y_true = np.asarray(y_true)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    within = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    actual_coverage = float(np.mean(within))

    return {
        'actual_coverage':     actual_coverage,
        'expected_coverage':   confidence_level,
        'mean_interval_width': float(np.mean(upper_bounds - lower_bounds)),
        'coverage_difference': actual_coverage - confidence_level,
    }

def hexaclus_pairing(results: Dict[str, Dict[str, Any]],
                     kernel_to_baseline: Optional[Dict[str, str]] = None,
                     prefix: str = 'HexaClus++'
                     ) -> Tuple[List[str], Dict[str, str]]:
   
    kernel_to_baseline = {'gaussian': 'gp', **(kernel_to_baseline or {})}
    disp_prefix = f'{prefix} ('  

    def _kernel_of(key: str) -> Optional[str]:

        if key.startswith('clustering_'):
            return key[len('clustering_'):]
        if key.startswith(disp_prefix) and key.endswith(')'):
            return key[len(disp_prefix):-1]
        return None

    order: List[str] = []
    rename: Dict[str, str] = {}
    used: set = set()

    for key in results:
        kernel = _kernel_of(key)
        if kernel is None:
            continue
        disp = f'{prefix} ({kernel})'
        if disp != key:                      # only rename raw 'clustering_*' keys
            rename[key] = disp
        order.append(key)
        used.add(key)
        base = kernel_to_baseline.get(kernel, kernel)
        if base in results and base not in used:
            order.append(base)
            used.add(base)

    # append any remaining models (baselines without a clustering counterpart)
    order += [k for k in results if k not in used]
    return order, rename


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
        # convert predictions and labels to original scale 
       
        if has_scaler:
            y_eval = inverse_transform_label(y_norm,    label_scaler, label_cols)
            pred_eval = inverse_transform_label(pred_norm, label_scaler, label_cols)
            if std_norm is not None:
                std_eval = inverse_transform_std(pred_norm, std_norm,
                                                   label_scaler, label_cols)
                lower_eval, upper_eval = inverse_transform_interval_bounds(
                    pred_norm, std_norm, label_scaler, label_cols, n_sigma=z_score
                )
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

        if std_eval is not None:
            interval_metrics = evaluate_prediction_intervals(
                y_eval, lower_eval, upper_eval, confidence_level=confidence_level
            )
            calib = evaluate_calibration(y_eval, pred_eval, std_eval, n_bins=10)

            interval_width = interval_metrics['mean_interval_width']
            coverage = interval_metrics['actual_coverage']
            ece = calib['ece']
            correlation = calib['correlation']
            p_value = calib['p_value']
            sharpness = calib['sharpness']
        else:
            interval_width = coverage = None
            ece = correlation = p_value = sharpness = None

        def fmt(v, decimals=4):
            return f"{v:.{decimals}f}" if v is not None else "N/A"

        scale_note = "(orig.)" if has_scaler else "(norm.)"
        tuning_time = result.get('tuning_time_s')
        n_configs = result.get('n_configs')
        time_per = result.get('time_per_config_s')

        rows.append({
            'Model': model_name,
            f'RMSE {scale_note}': fmt(rmse),
            f'MAE {scale_note}': fmt(mae),
            f'Sharpness {scale_note}': fmt(sharpness),
            f'Coverage ({int(confidence_level*100)}%)': fmt(coverage),
            f'ECE {scale_note}': fmt(ece),
            'Calib. Corr': fmt(correlation),
            'Tuning Time (s)': fmt(tuning_time, decimals=1),
            'N Configs': str(n_configs) if n_configs is not None else 'N/A',
            'Time/Config (s)': fmt(time_per, decimals=1),
        })

    df = pd.DataFrame(rows).set_index('Model')

    if order is not None:
        present = [m for m in order if m in df.index]
        rest = [m for m in df.index if m not in present]  # keep un-listed models at the end
        df = df.reindex(present + rest)
    if rename is not None:
        df = df.rename(index=rename)

    return df
