import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point
from evaluation import evaluate_calibration
from load_data import inverse_transform_label, inverse_transform_std
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_uncertainty_map_with_hatching(
    clustering_model: Any,
    test_instances: gpd.GeoDataFrame,
    y_pred: np.ndarray,
    std_pred: np.ndarray,
    uncertainty_threshold: float = 0.5,
    cmap: str = 'RdYlBu_r',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    title: str = "Prediction Map with Uncertainty Overlay",
    label_scaler=None,
    label_cols: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create map with predictions as colors and high-uncertainty regions marked with hatching.
    """
    has_scaler = (label_scaler is not None) and (label_cols is not None)
    # Convert to original scale before aggregation when scaler is provided
    y_pred = np.asarray(y_pred)
    std_pred = np.asarray(std_pred)
    if has_scaler:
        y_pred_plot = inverse_transform_label(y_pred, label_scaler, label_cols)
        std_pred_plot = inverse_transform_std(y_pred, std_pred, label_scaler, label_cols)
    else:
        y_pred_plot = y_pred
        std_pred_plot = std_pred

    # Get active polygons
    active_polygons, active_indices = clustering_model.get_active_polygons()

    # Aggregate by polygon
    polygon_preds = []
    polygon_stds = []
    polygon_counts = []
    polygon_geometries = []

    for poly_idx in active_indices:
        instance_indices = []
        for i, point in enumerate(test_instances.geometry):
            if active_polygons[active_indices.index(poly_idx)].contains(point):
                instance_indices.append(i)

        if len(instance_indices) > 0:
            poly_pred = np.mean([y_pred_plot[i] for i in instance_indices])
            poly_std = np.mean([std_pred_plot[i] for i in instance_indices])

            polygon_preds.append(poly_pred)
            polygon_stds.append(poly_std)
            polygon_counts.append(len(instance_indices))
            polygon_geometries.append(active_polygons[active_indices.index(poly_idx)])

    polygon_preds = np.array(polygon_preds)
    polygon_stds = np.array(polygon_stds)

    # Identify high uncertainty regions
    std_threshold = np.percentile(polygon_stds, uncertainty_threshold * 100)
    high_uncertainty = polygon_stds >= std_threshold

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': polygon_geometries,
        'prediction': polygon_preds,
        'uncertainty': polygon_stds,
        'high_uncert': high_uncertainty,
        'density': polygon_counts
    })

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Labels depend on whether we are in original or normalized space
    _pred_label = 'Mean Predicted Value (price/sqm)' if has_scaler else 'Mean Predicted Value'
    _std_label  = 'Std Dev (price/sqm)' if has_scaler else 'Std Dev'

    # Plot predictions
    gdf.plot(column='prediction', ax=ax, cmap=cmap,
             edgecolor='black', linewidth=0.5, legend=True,
             legend_kwds={'label': _pred_label,
                          'shrink': 0.4, 'pad': 0.01})
    # Access the colorbar and change font sizes
    cbar = ax.get_figure().axes[-1]  # colorbar axis
    cbar.set_ylabel(_pred_label, fontsize=17)
    # Add hatching for high uncertainty regions
    high_uncert_gdf = gdf[gdf['high_uncert']]
    if len(high_uncert_gdf) > 0:
        high_uncert_gdf.plot(ax=ax, facecolor='none', edgecolor='red',
                             linewidth=1.5, hatch='///', alpha=0.3,
                             label=f'High Uncertainty (>{std_threshold:.2f} {_std_label})')

    # Add instance counts as text
    for _, row in gdf.iterrows():
        centroid = row['geometry'].centroid
        ax.annotate(str(int(row['density'])),
                   xy=(centroid.x, centroid.y),
                   ha='center', va='center',
                   fontsize=7 + 3, color='black',
                   bbox=dict(boxstyle='circle', facecolor='white',
                           edgecolor='gray', alpha=0.7, pad=0.2))

    ax.set_title(title, fontsize=20 + 5, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10 + 5, framealpha=0.5)

    # Add statistics
    stats_text = (
        f"Total Polygons: {len(polygon_preds)}\n"
        f"High Uncertainty: {high_uncertainty.sum()} ({high_uncertainty.sum()/len(high_uncertainty)*100:.1f}%)\n"
        f"{_std_label} Range: [{polygon_stds.min():.2f}, {polygon_stds.max():.2f}]"
    )
    ax.text(
        0.02, 1.1, stats_text,
        transform=ax.transAxes,
        fontsize=16,
        # 'top', 'bottom', 'center', 'baseline', 'center_baseline'
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hatched uncertainty map to {save_path}")
    return fig, ax

# Feature importance 

def _prepare_polygon_data(
    clustering_model: Any,
    test_instances: gpd.GeoDataFrame
) -> Tuple[List[Dict], List[str], List[int]]:
    active_polygons, active_indices = clustering_model.get_active_polygons()
    feature_names = [c for c in test_instances.columns if c not in ['geometry', 'label']]

    polygon_data = []
    poly_indices = []

    for i, poly_idx in enumerate(active_indices):
        poly = active_polygons[i]
        mask = test_instances.geometry.apply(lambda pt: poly.contains(pt))
        poly_instances = test_instances[mask]

        if len(poly_instances) > 0:
            X = poly_instances[feature_names].values
            y = poly_instances['label'].values
            model = clustering_model.models[poly_idx]
            polygon_data.append({'X': X, 'y': y, 'model': model, 'poly_idx': poly_idx})
            poly_indices.append(poly_idx)

    return polygon_data, feature_names, poly_indices


def _compute_mse(polygon_data_list: List[Dict], label_scaler=None, label_cols=None) -> float:
    """Compute MSE across all polygons, optionally in original price/sqm scale."""
    from sklearn.metrics import mean_squared_error
    all_y = []
    all_pred = []
    for data in polygon_data_list:
        pred, _ = data['model'].predict(data['X'], return_std=True)
        all_y.extend(data['y'])
        all_pred.extend(pred)
    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    if label_scaler is not None and label_cols is not None:
        all_y = inverse_transform_label(all_y, label_scaler, label_cols)
        all_pred = inverse_transform_label(all_pred, label_scaler, label_cols)
    return mean_squared_error(all_y, all_pred)


def compute_feature_importance(
    clustering_model: Any,
    test_instances: gpd.GeoDataFrame,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
    label_scaler=None,
    label_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    polygon_data, feature_names, _ = _prepare_polygon_data(clustering_model, test_instances)
    # The unit is different for newyork sqft, have already corrected in final plot 
    _scale_unit = '(price/sqm)²' if label_scaler is not None and label_cols is not None else '(normalized)' 
    # Baseline score
    baseline_score = _compute_mse(polygon_data, label_scaler, label_cols)
    print(f"Baseline MSE {_scale_unit}: {baseline_score:.4f}")
    # Compute permutation importance for each feature
    print("\nComputing individual feature importance...")
    importances = []
    rng = np.random.RandomState(random_state)

    for feat_idx, feat_name in enumerate(feature_names):
        feat_importances = []
        for _ in range(n_repeats):
            # Create permuted data
            permuted_data = []
            for data in polygon_data:
                X_permuted = data['X'].copy()
                X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
                permuted_data.append({'X': X_permuted, 'y': data['y'], 'model': data['model']})

            # Compute score with permuted feature
            permuted_score = _compute_mse(permuted_data, label_scaler, label_cols)
            importance = permuted_score - baseline_score  # Higher = more important
            feat_importances.append(importance)
        importances.append({
            'feature': feat_name,
            'importance_mean': np.mean(feat_importances),
            'importance_std': np.std(feat_importances),
            'importance_abs': np.abs(np.mean(feat_importances))
        })
        print(f"  {feat_name}: {np.mean(feat_importances):.4f} ± {np.std(feat_importances):.4f} {_scale_unit}")
    # Create individual importance dataframe
    individual_df = pd.DataFrame(importances).sort_values('importance_abs', ascending=False)
    # Compute grouped importance if groups are provided
    grouped_df = None
    if feature_groups:
        print("\nComputing grouped feature importance...")
        grouped_importances = []

        for group_name, group_features in feature_groups.items():
            # Get indices of features in this group
            group_indices = [feature_names.index(f) for f in group_features if f in feature_names]

            if len(group_indices) == 0:
                print(f"  Warning: No features found for group '{group_name}'")
                continue

            group_imps = []
            rng_group = np.random.RandomState(random_state)
            for _ in range(n_repeats):
                # Permute all features in this group
                permuted_data = []
                for data in polygon_data:
                    X_permuted = data['X'].copy()
                    for idx in group_indices:
                        X_permuted[:, idx] = rng_group.permutation(X_permuted[:, idx])
                    permuted_data.append({'X': X_permuted, 'y': data['y'], 'model': data['model']})

                # Compute score with permuted group
                permuted_score = _compute_mse(permuted_data, label_scaler, label_cols)
                importance = permuted_score - baseline_score
                group_imps.append(importance)

            grouped_importances.append({
                'group': group_name,
                'n_features': len(group_indices),
                'importance_mean': np.mean(group_imps),
                'importance_std': np.std(group_imps),
                'importance_abs': np.abs(np.mean(group_imps))
            })
            print(f"  {group_name} ({len(group_indices)} features): "
                  f"{np.mean(group_imps):.4f} ± {np.std(group_imps):.4f} {_scale_unit}")

        grouped_df = pd.DataFrame(grouped_importances).sort_values('importance_abs', ascending=False)

    return individual_df, grouped_df


def plot_feature_importance(
    individual_df: pd.DataFrame,
    grouped_df: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> Tuple[plt.Figure, plt.Axes]:
    
    if grouped_df is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax2 = None

    # Plot individual feature importance
    top_features = individual_df.head(top_n)

    ax1.barh(
        range(len(top_features)),
        top_features['importance_mean'],
        xerr=top_features['importance_std'],
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.xaxis.get_offset_text().set_fontsize(20)
    ax1.set_xlabel('Importance (Mean ± Std)', fontsize=20)
    ax1.set_title(f'Top {top_n} Individual Features' + title, fontsize=20 + 5, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Plot grouped feature importance if available
    if grouped_df is not None and ax2 is not None:
        ax2.barh(
            range(len(grouped_df)),
            grouped_df['importance_mean'],
            xerr=grouped_df['importance_std'],
            color='coral',
            alpha=0.7,
            edgecolor='black'
        )
        ax2.set_yticks(range(len(grouped_df)))
        ax2.set_yticklabels([f"{row['group']} ({row['n_features']})"
                             for _, row in grouped_df.iterrows()], fontsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.xaxis.get_offset_text().set_fontsize(20)
        ax2.set_xlabel('Importance (Mean ± Std)', fontsize=20)
        ax2.set_title('Feature Group Importance' + title, fontsize=20 + 5, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    return fig, (ax1, ax2) if ax2 else (ax1,)


# Regional feature importance and plotting
def compute_regional_feature_importance(
    clustering_model: Any,
    test_instances: gpd.GeoDataFrame,
    feature_groups: Dict[str, List[str]],
    n_repeats: int = 5,
    random_state: int = 42,
    label_scaler=None,
    label_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    has_scaler = (label_scaler is not None) and (label_cols is not None) # always reverse back to original scale
    polygon_data, feature_names, _ = _prepare_polygon_data(clustering_model, test_instances)

    results = []
    rng = np.random.RandomState(random_state)

    for data in polygon_data:
        poly_idx = data['poly_idx']
        X, y, model = data['X'], data['y'], data['model']

        # Baseline prediction — convert to original scale when scaler is available
        pred, _ = model.predict(X, return_std=True)
        if has_scaler:
            y_eval    = inverse_transform_label(np.asarray(y),    label_scaler, label_cols)
            pred_eval = inverse_transform_label(np.asarray(pred), label_scaler, label_cols)
        else:
            y_eval, pred_eval = np.asarray(y), np.asarray(pred)
        baseline_mse = np.mean((y_eval - pred_eval) ** 2)
        mean_pred = np.mean(pred_eval)

        # Compute importance for each feature group
        for group_name, group_features in feature_groups.items():
            group_indices = [feature_names.index(f) for f in group_features if f in feature_names]

            if len(group_indices) == 0:
                continue

            importance_scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                for idx in group_indices:
                    X_permuted[:, idx] = rng.permutation(X_permuted[:, idx])

                pred_perm, _ = model.predict(X_permuted, return_std=True)
                if has_scaler:
                    pred_perm_eval = inverse_transform_label(np.asarray(pred_perm), label_scaler, label_cols)
                else:
                    pred_perm_eval = np.asarray(pred_perm)
                permuted_mse = np.mean((y_eval - pred_perm_eval) ** 2)
                importance_scores.append(permuted_mse - baseline_mse)

            results.append({
                'poly_idx': poly_idx,
                'group': group_name,
                'importance_mean': np.mean(importance_scores),
                'importance_std': np.std(importance_scores),
                'importance_abs': np.abs(np.mean(importance_scores)),
                'mean_pred': mean_pred,
                'n_instances': len(y)
            })

    return pd.DataFrame(results)


def plot_regional_feature_importance(
    clustering_model: Any,
    test_instances: gpd.GeoDataFrame,
    feature_groups: Dict[str, List[str]],
    y_pred: Optional[np.ndarray] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    cmap: str = 'RdYlBu_r',
    figsize: Tuple[int, int] = (16, 12),
    bar_size: Optional[Tuple[float, float]] = None,
    palette: str = 'husl',
    save_path: Optional[str] = None,
    title: str = "Regional Feature Importance Map",
    label_scaler=None,
    label_cols: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:

    has_scaler = (label_scaler is not None) and (label_cols is not None)

    # Convert y_pred to original scale before aggregation when scaler is available
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        if has_scaler:
            y_pred_plot = inverse_transform_label(y_pred, label_scaler, label_cols)
        else:
            y_pred_plot = y_pred
    else:
        y_pred_plot = None

    # Compute regional importance (in original scale if scaler provided)
    print("Computing regional feature importance...")
    regional_df = compute_regional_feature_importance(
        clustering_model, test_instances, feature_groups, n_repeats, random_state,
        label_scaler, label_cols
    )

    if len(regional_df) == 0:
        raise ValueError("No polygons with sufficient test instances found")

    # Get active polygons
    active_polygons, active_indices = clustering_model.get_active_polygons()
    poly_to_geom = {idx: active_polygons[i] for i, idx in enumerate(active_indices)}

    # Get unique polygons with data
    poly_indices = regional_df['poly_idx'].unique()
    group_names = list(feature_groups.keys())
    n_groups = len(group_names)

    # Get colors from seaborn palette
    colors = sns.color_palette(palette, n_colors=n_groups)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get prediction range for colormap
    # If y_pred_plot is provided, aggregate by polygon (same as plot_uncertainty_map_with_hatching)
    # y_pred_plot is already in original scale when has_scaler, otherwise normalized.
    if y_pred_plot is not None:
        pred_by_poly = {}
        for poly_idx in poly_indices:
            if poly_idx not in poly_to_geom:
                continue
            poly = poly_to_geom[poly_idx]
            instance_indices = [i for i, point in enumerate(test_instances.geometry)
                                if poly.contains(point)]
            if len(instance_indices) > 0:
                pred_by_poly[poly_idx] = np.mean([y_pred_plot[i] for i in instance_indices])
        pred_by_poly = pd.Series(pred_by_poly)
    else:
        # Fall back to internally computed predictions (already in original scale if has_scaler)
        pred_by_poly = regional_df.groupby('poly_idx')['mean_pred'].first()

    norm = Normalize(vmin=pred_by_poly.min(), vmax=pred_by_poly.max())
    colormap = cm.get_cmap(cmap)

    # Calculate fixed bar size based on map extent if not provided
    if bar_size is None:
        all_bounds = [poly_to_geom[idx].bounds for idx in poly_indices if idx in poly_to_geom]
        min_x = min(b[0] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        map_width = max_x - min_x
        map_height = max_y - min_y
        bar_width = map_width * 0.04
        bar_height = map_height * 0.015
    else:
        bar_width, bar_height = bar_size

    for poly_idx in poly_indices:
        if poly_idx not in poly_to_geom:
            continue

        geom = poly_to_geom[poly_idx]
        pred = pred_by_poly[poly_idx]
        color = colormap(norm(pred))

        # Plot polygon
        gpd.GeoDataFrame([{'geometry': geom}]).plot(
            ax=ax, color=color, edgecolor='black', linewidth=0.5, zorder=1
        )

    # Calculate bar offset (slightly above centroid)
    bar_offset_y = bar_height * 1.5

    for poly_idx in poly_indices:
        if poly_idx not in poly_to_geom:
            continue

        geom = poly_to_geom[poly_idx]

        # Get importance values for this polygon
        poly_data = regional_df[regional_df['poly_idx'] == poly_idx]
        values = [poly_data[poly_data['group'] == g]['importance_mean'].values[0]
                  if len(poly_data[poly_data['group'] == g]) > 0 else 0
                  for g in group_names]

        # Clamp negatives and compute percentages
        values = [max(0, v) for v in values]
        total = sum(values)
        if total > 0:
            percentages = [v / total for v in values]
        else:
            percentages = [1.0 / n_groups] * n_groups  # Equal split if no importance

        # Draw horizontal stacked bar offset above centroid
        centroid = geom.centroid
        bar_x = centroid.x - bar_width / 2
        bar_y = centroid.y + bar_offset_y

        # Draw small connecting line from centroid to bar
        ax.plot([centroid.x, centroid.x], [centroid.y, bar_y],
                color='gray', linewidth=0.8, zorder=2, alpha=0.7)
        ax.plot(centroid.x, centroid.y, 'o', color='gray', markersize=2, zorder=2)

        # Draw stacked segments with high zorder
        current_x = bar_x
        for pct, bcolor in zip(percentages, colors):
            segment_width = pct * bar_width
            if segment_width > 0:
                rect = Rectangle(
                    (current_x, bar_y),
                    segment_width, bar_height,
                    facecolor=bcolor, edgecolor='white', linewidth=0.3, alpha=0.9,
                    zorder=3
                )
                ax.add_patch(rect)
                current_x += segment_width

        # Add black border around entire bar
        border = Rectangle(
            (bar_x, bar_y), bar_width, bar_height,
            facecolor='none', edgecolor='black', linewidth=0.5, zorder=4
        )
        ax.add_patch(border)

    # Expand axis limits to ensure bars are fully visible
    y_lim = ax.get_ylim()
    y_margin = (y_lim[1] - y_lim[0]) * 0.05
    ax.set_ylim(y_lim[0], y_lim[1] + y_margin + bar_height + bar_offset_y)

    # Add colorbar for predictions
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
    _cbar_label = 'Mean Predicted Value (price/sqm)' if has_scaler else 'Mean Predicted Value'
    cbar.set_label(_cbar_label, fontsize=12 + 5)

    # Add legend for feature groups
    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black')
                      for i in range(n_groups)]
    ax.legend(legend_handles, group_names, loc='upper left', fontsize=15 + 5,
              title='Feature Groups (%)', framealpha=0.9)

    ax.set_title(title, fontsize=20 + 5, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved regional feature importance map to {save_path}")

    return fig, ax

# Plot calibration comparison curves across models
def plot_calibration_comparison(
    results: Dict[str, Dict[str, Any]],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    title: str = "Calibration Comparison Across Models",
    label_scaler=None,
    label_cols: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    
    has_scaler = (label_scaler is not None) and (label_cols is not None)

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    max_val = 0  # Track maximum value for perfect calibration line

    # Plot each model calibration curve
    for (model_name, result), color in zip(results.items(), colors):
        # Skip models without uncertainty estimates
        if result.get('test_std') is None:
            print(f"Skipping {model_name}: no uncertainty estimates")
            continue

        y_true_norm = np.array(result['test_y'])
        pred_norm = np.array(result['test_pred'])
        std_norm = np.array(result['test_std'])

        # Convert to original scale when scaler is available
        if has_scaler:
            y_true        = inverse_transform_label(y_true_norm, label_scaler, label_cols)
            y_pred        = inverse_transform_label(pred_norm,   label_scaler, label_cols)
            uncertainties = inverse_transform_std(pred_norm, std_norm, label_scaler, label_cols)
        else:
            y_true = y_true_norm
            y_pred = pred_norm
            uncertainties = std_norm

        # Compute calibration metrics using existing function
        calibration_metrics = evaluate_calibration(
            y_true=y_true,
            y_pred=y_pred,
            uncertainties=uncertainties,
            n_bins=n_bins,
            save_path=None,
            show_plot=False
        )

        bin_means_uncertainty = calibration_metrics['bin_means_uncertainty']
        bin_means_error = calibration_metrics['bin_means_error']
        ece = calibration_metrics['ece']
        correlation = calibration_metrics['correlation']

        # Update max value
        max_val = max(max_val, max(bin_means_uncertainty), max(bin_means_error))

        # Plot calibration curve
        ax.plot(
            bin_means_uncertainty,
            bin_means_error,
            marker='o',
            linewidth=2,
            markersize=8,
            color=color,
            label=f"{model_name.upper()} (ECE={ece:.3f}, r={correlation:.2f})",
            alpha=0.8
        )

    # Plot perfect calibration line
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2.5, label='Perfect Calibration', zorder=0)

    # Styling
    ax.set_xlabel('Predicted Uncertainty (Std Dev)', fontsize=14 + 5, fontweight='bold')
    ax.set_ylabel('Observed Error (MAE)', fontsize=14 + 5, fontweight='bold')
    ax.set_title(title, fontsize=16 + 5, fontweight='bold', pad=20)
    ax.legend(fontsize=11 + 5, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    # Add text box with interpretation guide
    guide_text = (
        "Closer to diagonal = better calibrated\n"
        "Lower ECE = better calibration\n"
        "Higher r = better correlation"
    )
    ax.text(
        0.98, 0.02, guide_text,
        transform=ax.transAxes,
        fontsize=12 + 5,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration comparison plot to {save_path}")

    return fig, ax

