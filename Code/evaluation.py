

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from typing import Optional, Dict, Any, List

from utils import gaussian_nll
from load_data import (
    inverse_transform_label,
    inverse_transform_std,
    inverse_transform_interval_bounds, # inverse transform for intervals in original scale
)

# Calibration
def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainties: np.ndarray,
                         n_bins: int = 10,
                         save_path: Optional[str] = None,
                         show_plot: bool = False) -> Dict[str, Any]:
    """
    Evaluate calibration of uncertainty estimates.
    Bins predictions by predicted uncertainty and checks whether the mean predicted std tracks the mean absolute error in each bin.  
    All inputs have already been in the target scale (original).
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

    for i in range(n_bins):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_uncertainties)
        bin_means_uncertainty.append(np.mean(sorted_uncertainties[start:end]))
        bin_means_error.append(np.mean(sorted_errors[start:end]))
        bin_stds_error.append(np.std(sorted_errors[start:end]))

    correlation, p_value = stats.pearsonr(bin_means_uncertainty, bin_means_error)
    ece = np.mean(np.abs(np.array(bin_means_uncertainty) - np.array(bin_means_error)))
    sharpness = np.mean(uncertainties)   # mean predicted std in the given scale

    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.errorbar(bin_means_uncertainty, bin_means_error, yerr=bin_stds_error,
                    fmt='o-', linewidth=2, markersize=8, capsize=5, label='Observed')
        max_val = max(max(bin_means_uncertainty), max(bin_means_error))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
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


# Prediction interval evaluation

def evaluate_prediction_intervals(y_true: np.ndarray,
                                  lower_bounds: np.ndarray,
                                  upper_bounds: np.ndarray,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Evaluate prediction interval coverage and width. We didnt really report this in the paper, but it can be useful for deeper analysis of uncertainty quality.
    """
    y_true       = np.asarray(y_true)
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

def print_comparison_results(results: Dict[str, Dict[str, Any]],
                             label_scaler=None,
                             label_cols: Optional[List[str]] = None,
                             confidence_level: float = 0.95) -> pd.DataFrame:

    has_scaler = (label_scaler is not None) and (label_cols is not None)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    rows = []

    for model_name, result in results.items():
        y_norm = np.asarray(result['test_y'])
        pred_norm = np.asarray(result['test_pred'])
        std_norm  = result['test_std']
        std_norm  = np.asarray(std_norm) if std_norm is not None else None

        # Convert predictions and labels to original scale (if scaler given)
        if has_scaler:
            y_eval = inverse_transform_label(y_norm,    label_scaler, label_cols)
            pred_eval = inverse_transform_label(pred_norm, label_scaler, label_cols)
            if std_norm is not None:
                std_eval = inverse_transform_std(pred_norm, std_norm, label_scaler, label_cols)
                lower_eval, upper_eval = inverse_transform_interval_bounds(pred_norm, std_norm, label_scaler, label_cols, n_sigma=z_score)
            else:
                std_eval = lower_eval = upper_eval = None
        else:
            # Fallback to normalized scale 
            print("Warning: No scaler provided, evaluating metrics in normalized scale.")
            y_eval = y_norm
            pred_eval = pred_norm
            std_eval = std_norm
            if std_norm is not None:
                lower_eval = pred_norm - z_score * std_norm
                upper_eval = pred_norm + z_score * std_norm
            else:
                lower_eval = upper_eval = None
        # Evaluation metrics
        rmse = float(np.sqrt(mean_squared_error(y_eval, pred_eval)))
        r2 = float(r2_score(y_eval, pred_eval))
        nll = float(gaussian_nll(y_norm, pred_norm, std_norm)) if std_norm is not None else None
        if std_eval is not None:
            interval_metrics = evaluate_prediction_intervals(y_eval, lower_eval, upper_eval, confidence_level=confidence_level)
            calib = evaluate_calibration(y_eval, pred_eval, std_eval, n_bins=10)
            mean_std = float(np.mean(std_eval))
            interval_width = interval_metrics['mean_interval_width']
            coverage = interval_metrics['actual_coverage']
            ece = calib['ece']
            correlation = calib['correlation']
            p_value = calib['p_value']
            sharpness = calib['sharpness']
        else:
            mean_std = interval_width = coverage = None
            ece = correlation = p_value = sharpness = None
        # format
        def _fmt(v, decimals=4):
            return f"{v:.{decimals}f}" if v is not None else "N/A"
        scale_note = "(orig.)" if has_scaler else "(norm.)"
        rows.append({
            'Model': model_name,
            f'RMSE {scale_note}': _fmt(rmse),
            'R2':  _fmt(r2),
            'NLL (norm.)': _fmt(nll),
            f'Mean Std {scale_note}': _fmt(mean_std),
            f'Sharpness {scale_note}': _fmt(sharpness),
            f'95% PI Width {scale_note}': _fmt(interval_width),
            f'Coverage ({int(confidence_level*100)}%)': _fmt(coverage),
            f'ECE {scale_note}': _fmt(ece),
            'Calib. Corr': _fmt(correlation), # not reported 
            'Calib. p-value': _fmt(p_value, decimals=4), # not reported
        })
    df = pd.DataFrame(rows).set_index('Model')
    return df
