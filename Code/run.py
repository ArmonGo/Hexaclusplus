import os
import warnings
import numpy as np
from cv import GridSearcher
from load_data import load_london, load_gdf, load_paris, load_newyork
from baseline_models import tune_and_evaluate_all_baselines
import pickle

citys = [ 'new_york', 
         'paris', 'london']
selected_areas = [ "New York City, US", 
    "Paris, FR", "London, UK"]
load_fs = [ load_newyork, 
    load_paris, load_london]

max_iter = 3000
patience = 20
save_path = './results/'

max_iter = 3000
patience = 20
save_path = './results/'
N_JOBS = -1   # grid-level parallelism: 1 = sequential; -1 = all cores; N = N processes


param_grid = {
    'resolutions': [7, 8, 9],   
    'use_simulated_annealing': [True],
    'initial_temp': [1.96],  
    'cooling_rate': [None],
    'min_samples_per_hexagon': [30],
    'scoring_method': ['beta_nll'],    
    'beta': [0.5], 
    'halo_buffer': [0], 
    'random_seed': [0], 
}


kernel_grids = {
    'bayesian': {
        'alpha':  [1e-4, 1e-2],  
        'lambda': [1e-4, 1e-2],   
        'max_iter': [50, 100, 200, 300],  # update min to 
        'tol':    [1e-2, 1e-3, 1e-4],  
    },
    'knn': {
        'n_neighbors': [3, 5, 8, 10, 15],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators':     [50],  
        'max_depth':         [2, 3, 4],  
        'min_samples_leaf': [2, 5],
        'trees_growth':     ['sqrt'],
        'max_estimators':   [300]
    },
    'gaussian': {
        'optimizer':    ['fmin_l_bfgs_b'],   # use default
        'length_scale': [0.5, 1.0],
        'noise_level':  [1e-3],
    }
}


def load_data(l_f):
    # load data 
    df , label_scaler, label_cols = l_f( split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = False)
    df = df.reset_index(drop=True)
    (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test) = load_gdf(df)
    return (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test), label_scaler, label_cols

def run_baselines(df_train, df_val, df_test, save_path, beta):
    baseline_results = tune_and_evaluate_all_baselines(
    train_instances = df_train,
    val_instances=df_val,
    test_instances=df_test,
    target_col= 'label',
    models_to_run=['grfr', 'krigingrf', 'bayesian', 'kriging', 'gp', 'knn', 'rf'],
    verbose = True,
    save_path=save_path,
    metric='beta_nll',  
    beta=beta)  
    return baseline_results

def run_clustering(save_path, param_grid, kernel_grids, gdf_train, gdf_val, gdf_test, area, max_iter, patience):
    
    results = {}

    for kernel_name, kernel_param_grid in kernel_grids.items():
        print(f"\n{'='*70}")
        print(f"RUNNING CLUSTERING WITH KERNEL: {kernel_name.upper()}")
        print(f"{'='*70}\n")
        current_param_grid = {**param_grid, 'kernel': [kernel_name]}
        searcher = GridSearcher(
            grid=current_param_grid,
            save_path=save_path,
            kernel_grid={kernel_name: kernel_param_grid}
        )

        print(f"Kernel: {kernel_name}")
        print(f"Kernel hyperparameters: {kernel_param_grid}")
        print("Starting hyperparameter search...")

        best_model, _, test_preds, test_instances, test_uncertainties, n_configs, tuning_time = searcher.cv_clustering(
            gdf_train=gdf_train,
            gdf_val=gdf_val,
            gdf_test=gdf_test,
            max_iter=max_iter,
            patience=patience,
            selected_area=area,
            gdf_test_orig=gdf_test,
        )
        result_key = f'clustering_{kernel_name}'
        results[result_key] = {
            'model': best_model,
            'test_pred': test_preds,
            'test_std': test_uncertainties,
            'test_y': test_instances.label,
            # 'temperature': temperature,
            'test_instance': test_instances,
            'best_params': searcher.best_param,
            'kernel': kernel_name,
            'tuning_time_s': tuning_time,
            'n_configs': n_configs,
            'time_per_config_s': tuning_time / n_configs,
        }

        with open(os.path.join(save_path, "clustering_results_partial.pkl"), "wb") as f:
            pickle.dump(results, f)

        print(f"\n{kernel_name.upper()} Results:")
        print(f"  Best params: {searcher.best_param}")
    return results

def main(param_grid, kernel_grids, citys, selected_areas, max_iter, patience):
    for l in range(len(citys)):
        save_path = './results/' + citys[l] + '/'
        os.makedirs(save_path, exist_ok=True)
        l_f = load_fs[l]
        area = selected_areas[l]

        print(f"\n{'#'*70}")
        print(f"PROCESSING CITY: {citys[l].upper()}")
        print(f"{'#'*70}\n")

        (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test), label_scaler, label_cols = load_data(l_f)
        
        if citys[l] in ['paris', 'london']:
            param_grid['resolutions'] = [8, 9, 10] # add more resolution to smaller area
        clustering_results = run_clustering(
            save_path, param_grid, kernel_grids,
            gdf_train, gdf_val, gdf_test, area, max_iter, patience
        )
        baseline_results = run_baselines(df_train, df_val, df_test, save_path, beta=param_grid['beta'][0])  # Use the first beta value for baseline tuning
        with open(save_path + "results.pkl", "wb") as f:
            pickle.dump(baseline_results, f)

        clustering_results.update(baseline_results)

        with open(save_path + "results.pkl", "wb") as f:
            pickle.dump(clustering_results, f)

        print(f"\nResults saved to {save_path}results.pkl")
        print(f"Models in results: {list(clustering_results.keys())}")


if __name__ == "__main__":
    main(param_grid, kernel_grids, citys, selected_areas, max_iter, patience)