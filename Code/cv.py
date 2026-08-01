from clustering import Clustering
import numpy as np
import copy
import time
from sklearn.model_selection import ParameterGrid
import pickle


# default hyperparameter grids for each kernel type
DEFAULT_KERNEL_GRIDS = {
    'bayesian': {
        'alpha_1': [1e-6, 1e-4, 1e-2],
        'alpha_2': [1e-6, 1e-4, 1e-2],
        'lambda_1': [1e-6, 1e-4, 1e-2],
        'lambda_2': [1e-6, 1e-4, 1e-2]
    },
    'knn': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    }
}


class GridSearcher:
    def __init__(self, grid, save_path, kernel_grid=None):
       
        self.param_grid = ParameterGrid(grid)
        self.kernel_grid = kernel_grid if kernel_grid is not None else DEFAULT_KERNEL_GRIDS
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None
        self.save_path = save_path

    def _get_kernel_param_combinations(self, kernel):
        if kernel in self.kernel_grid:
            return list(ParameterGrid(self.kernel_grid[kernel]))
        return [{}]  

    def cv_clustering(self, gdf_train, gdf_val, gdf_test, max_iter, patience,
                     selected_area='London, UK', gdf_test_orig=None):
        
        total_combinations = 0
        for param in self.param_grid:
            kernel = param.get('kernel', 'bayesian')
            total_combinations += len(self._get_kernel_param_combinations(kernel))

        print(f"\nGrid Search Configuration:")
        print(f"  Total parameter combinations: {total_combinations}\n")

        t_start = time.time()

        for param in self.param_grid:
            r = param.get('resolutions', 7)
            kernel = param.get('kernel', 'bayesian')
            uncertainty_weight = param.get('uncertainty_weight', 0.1)
            use_simulated_annealing = param.get('use_simulated_annealing', False)
            initial_temp = param.get('initial_temp', 1.0)
            cooling_rate = param.get('cooling_rate', 0.95)
            min_samples_per_hexagon = param.get('min_samples_per_hexagon', 20)
            scoring_method = param.get('scoring_method', 'beta_nll')
            beta = param.get('beta', 0.5)
            halo_buffer = param.get('halo_buffer', 0.0)
            random_seed = param.get('random_seed', 42)

            kernel_param_combinations = self._get_kernel_param_combinations(kernel)

            for kernel_params in kernel_param_combinations:
                current_config = {**param, 'kernel_params': kernel_params}
                print(f"Testing: kernel={kernel}, params={kernel_params}")

                cl = Clustering(gdf_train.copy(),
                                gdf_val.copy(),
                            save_path=self.save_path,
                            selected_area=selected_area,
                            resolution=r,
                            kernel=kernel,
                            kernel_params=kernel_params,
                            uncertainty_weight=uncertainty_weight,
                            min_samples_per_hexagon=min_samples_per_hexagon,
                            scoring_method=scoring_method,
                            beta=beta,   # only used when scoring_method == 'beta_nll'
                            min_polygons=10,
                            merge_threshold=0.001,   
                            halo_buffer=halo_buffer, 
                            random_seed=random_seed, 
                        )

                # Run clustering with consistent scoring method
                _, _ = cl.construct_clustering(
                    max_iter=max_iter,
                    patience=patience,
                    use_simulated_annealing=use_simulated_annealing,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate
                )

                # load best model
                b_m = cl.load_best_instance(self.save_path)

                # evaluate on validation set
                val_score, _, _, _ = b_m.validate()

                # check if this is the best model
                if val_score < self.best_score:
                    self.best_score = val_score
                    self.best_param = current_config
                    self.best_model = copy.deepcopy(b_m)
                    print(f"New best model! validation scores: {val_score:.4f}")

        # Final evaluation on test set
        print(f"\n{'='*70}")
        print(f"BEST MODEL FOUND")
        print(f"{'='*70}")
        print(f"Best {cl.scoring_method}: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_param}")

        test_data = gdf_test_orig if gdf_test_orig is not None else gdf_test
        total_time = time.time() - t_start
        score, preds, instances, uncertainties = self.best_model.predict(test_data)

        print(f"\nTest set performance:")
        print(f"  {cl.scoring_method}: {score:.4f}")
        print(f"  Mean uncertainty: {np.mean(uncertainties):.4f}")
        print(f"  Total tuning time: {total_time:.1f}s over {total_combinations} configs "
              f"({total_time/total_combinations:.1f}s/config)")

        return self.best_model, score, preds, instances, uncertainties, total_combinations, total_time
