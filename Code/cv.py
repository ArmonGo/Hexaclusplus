from clustering import Clustering
import numpy as np
import copy
from sklearn.model_selection import ParameterGrid


# Default hyperparameter grids for each kernel type
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
        """
        Grid search for clustering hyperparameters.
        """
        self.param_grid = ParameterGrid(grid)
        self.kernel_grid = kernel_grid if kernel_grid is not None else DEFAULT_KERNEL_GRIDS
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None
        self.save_path = save_path

    def _get_kernel_param_combinations(self, kernel):
        if kernel in self.kernel_grid:
            return list(ParameterGrid(self.kernel_grid[kernel]))
        return [{}]  # Empty dict if no kernel-specific params

    def cv_clustering(self, gdf_train, gdf_val, gdf_test, max_iter, patience,
                     selected_area='London, UK', gdf_test_orig=None):
        # Count total combinations
        total_combinations = 0
        for param in self.param_grid:
            kernel = param.get('kernel', 'bayesian')
            total_combinations += len(self._get_kernel_param_combinations(kernel))

        print(f"Grid Search Configuration:")
        print(f"Total parameter combinations: {total_combinations}\n")

        for param in self.param_grid:
            # Extract general parameters with defaults
            print('param', param)
            r = param.get('resolutions', 7)
            print('resolution:  -----------------', r)
            kernel = param.get('kernel', 'bayesian')
            uncertainty_weight = param.get('uncertainty_weight', 0.1)
            use_simulated_annealing = param.get('use_simulated_annealing', False)
            initial_temp = param.get('initial_temp', 1.0)
            cooling_rate = param.get('cooling_rate', 0.95)
            min_samples_per_hexagon = param.get('min_samples_per_hexagon', 20)
            scoring_method = param.get('scoring_method', 'nll')

            # Iterate over kernel-specific hyperparameters
            kernel_param_combinations = self._get_kernel_param_combinations(kernel)

            for kernel_params in kernel_param_combinations:
                current_config = {**param, 'kernel_params': kernel_params}
                print(f"Testing: kernel={kernel}, params={kernel_params}")

                # Create clustering instance
                cl = Clustering(gdf_train.copy(), gdf_val.copy(),
                            save_path=self.save_path,
                            selected_area=selected_area,
                            resolution=r,
                            kernel=kernel,
                            kernel_params=kernel_params,
                            uncertainty_weight=uncertainty_weight,
                            min_samples_per_hexagon=min_samples_per_hexagon,
                            scoring_method=scoring_method,
                            min_polygons=10,           # Keep at least 10 regions
                            merge_threshold=0.01     # Only merge if improvement > 0.01
                        )

                # Run clustering with consistent scoring method
                _, _, _ = cl.construct_clustering(
                    max_iter=max_iter,
                    patience=patience,
                    use_simulated_annealing=use_simulated_annealing,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate
                )

                # Load best model
                b_m = cl.load_best_instance(self.save_path)
                # Evaluate on validation set
                val_score, _, _, _ = b_m.validate()
                # Check if this is the best model
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
        score, preds, instances, uncertainties  = self.best_model.predict(test_data)
       
        print(f"\nTest set performance:")
        print(f"  {cl.scoring_method}: {score:.4f}")
        print(f"  Mean uncertainty: {np.mean(uncertainties):.4f}")

        return self.best_model, score, preds, instances, uncertainties
