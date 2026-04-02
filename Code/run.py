import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*sklearn.*parallel.*")

import numpy as np
from cv import GridSearcher
from load_data import load_london, load_gdf, load_paris, load_newyork
from baseline_models import tune_and_evaluate_all_baselines
from utils import gaussian_nll
import pickle

citys = ['new_york', 'paris', 'london']
selected_areas = ["New York City, US", "Paris, FR", "London, UK"]
load_fs = [load_newyork, load_paris, load_london]
max_iter = 3000
patience = 20 
save_path = './results/'

# Define general hyperparameters for clustering (shared across kernels)
param_grid = {
    'resolutions': [7, 8, 9],  # H3 resolution
    'uncertainty_weight': [None], # DOESNT MATTER BECAUSE WE ARE USING NLL
    'use_simulated_annealing': [True],
    'initial_temp': [1.0],
    'cooling_rate': [0.95],
    'min_samples_per_hexagon': [30],
    'scoring_method': ['nll']
}


# Define kernel-specific hyperparameter grids
kernel_grids = {
    'bayesian': {
        'alpha_1':  [1e-6, 1e-3, 1e0, 1e2],
        'alpha_2':  [1e-6, 1e-3, 1e0, 1e2],
        'lambda_1':  [1e-6, 1e-3, 1e0, 1e2],
        'lambda_2':  [1e-6, 1e-3, 1e0, 1e2]
    },
    'knn': {
        'n_neighbors': [3, 5, 8, 10],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators':     [50],
        'max_depth':        [2, 3, 4],   # shallow — local n is 30–200
        'min_samples_leaf': [2, 5]
    },
    'gaussian': {
        'length_scale': [0.5, 1.0],
        'noise_level':  [1e-3, 1e-2]
    }
}

def load_data(l_f):
    # load data 
    df , label_scaler, label_cols = l_f( split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = False)
    df = df.reset_index(drop=True)
    (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test) = load_gdf(df)
    return (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test), label_scaler, label_cols

def run_baselines(df_train, df_val, df_test):
    baseline_results = tune_and_evaluate_all_baselines(
    train_instances = df_train,
    val_instances=df_val,
    test_instances=df_test,
    target_col= 'label',
    models_to_run=['bayesian', 'kriging', 'gp', 'knn', 'rf'],  # [ 'bayesian', 'kriging', 'gp', 'knn', 'rf'],
    verbose = True)
    return baseline_results

def run_clustering(save_path, param_grid, kernel_grids, gdf_train, gdf_val, gdf_test, area, max_iter, patience):
    """
    Run clustering with different kernels and return results for each kernel.
    """
    results = {}

    # Loop over each kernel type
    for kernel_name, kernel_param_grid in kernel_grids.items():
        print(f"\n{'='*70}")
        print(f"RUNNING CLUSTERING WITH KERNEL: {kernel_name.upper()}")
        print(f"{'='*70}\n")

        # Add kernel to the general param_grid
        current_param_grid = {**param_grid, 'kernel': [kernel_name]}

        # Create grid searcher with kernel-specific hyperparameters
        searcher = GridSearcher(
            grid=current_param_grid,
            save_path=save_path,
            kernel_grid={kernel_name: kernel_param_grid}
        )

        print(f"Kernel: {kernel_name}")
        print(f"Kernel hyperparameters: {kernel_param_grid}")
        print("Starting hyperparameter search...")

        # Run grid search for this kernel
        best_model, _, test_preds, test_instances, test_uncertainties = searcher.cv_clustering(
            gdf_train=gdf_train,
            gdf_val=gdf_val,
            gdf_test=gdf_test,
            max_iter=max_iter,
            patience=patience,
            selected_area=area,
            gdf_test_orig=gdf_test,
        )



        # Store results with kernel name as key
        result_key = f'clustering_{kernel_name}'
        results[result_key] = {
            'model': best_model,
            'test_pred': test_preds,
            'test_std': test_uncertainties,
            'test_y': test_instances.label,
            'test_instance': test_instances,
            'best_params': searcher.best_param,
            'kernel': kernel_name
        }

        print(f"\n{kernel_name.upper()} Results:")
        print(f"  Best params: {searcher.best_param}")
    return results

def main(param_grid, kernel_grids, citys, selected_areas, max_iter, patience):
    # for l in range(len(citys)):
    for l in range(2,3,1):
        save_path = './results/' + citys[l] + '/'
        l_f = load_fs[l]
        area = selected_areas[l]

        print(f"\n{'#'*70}")
        print(f"PROCESSING CITY: {citys[l].upper()}")
        print(f"{'#'*70}\n")

        (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test), label_scaler, label_cols = load_data(l_f)

        # Collect baseline results
        
        baseline_results = run_baselines(df_train, df_val, df_test)
        with open(save_path + "results.pkl", "wb") as f:
            pickle.dump(baseline_results, f)

        # Run clustering with all kernels
        if citys[l] in ['paris', 'london']:
            param_grid['resolutions'] = [8, 9, 10] # add more resolution to smaller area 
        clustering_results = run_clustering(
            save_path, param_grid, kernel_grids,
            gdf_train, gdf_val, gdf_test, area, max_iter, patience
        )

        # Combine all results
        clustering_results.update(baseline_results)

        # Save results
        with open(save_path + "results.pkl", "wb") as f:
            pickle.dump(clustering_results, f)

        print(f"\nResults saved to {save_path}results.pkl")
        print(f"Models in results: {list(clustering_results.keys())}")


if __name__ == "__main__":
    main(param_grid, kernel_grids, citys, selected_areas, max_iter, patience)
