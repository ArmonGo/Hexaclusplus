
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
import pickle
import os
import time
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from pykrige.uk import UniversalKriging
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
from pandas import DataFrame
from utils import gaussian_beta_nll
from PyGRF import PyGRFBuilder

# beta for the 'beta_nll' tuning metric , by defalt 0.5
BETA_NLL_BETA = 0.5

def extract_coordinates(instances) -> np.ndarray:
    return np.array([[p.x, p.y] for p in instances.geometry])


class BaselineModel:
    """Base class for all baseline models."""

    def __init__(self, name: str, supports_uncertainty: bool = False):
        self.name = name
        self.supports_uncertainty = supports_uncertainty
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None,
                return_std: bool = False) -> Tuple:
        raise NotImplementedError

    def score(self,
               X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None,
               metric: str = 'rmse') -> float:
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before scoring")

        if metric == 'beta_nll' and not self.supports_uncertainty:
            raise ValueError(f"{self.name} does not support uncertainty metrics")

        if metric == 'beta_nll':
            preds, std = self.predict(X, coords, return_std=True)
            score = gaussian_beta_nll(y_true=y, y_mean=preds, y_std=std, beta=BETA_NLL_BETA)
            return score, preds, std
        else:  # rmse
            preds = self.predict(X, coords, return_std=False)
            return np.sqrt(mean_squared_error(y, preds)), preds


class GlobalBayesianRidge(BaselineModel):
    """Global Bayesian Ridge Regression (with defined uncertainty)."""

    def __init__(self, alpha_1: float = 1e-6, alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6, lambda_2: float = 1e-6):
        super().__init__(name="Global Bayesian Ridge", supports_uncertainty=True)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = BayesianRidge(
            alpha_1=alpha_1, alpha_2=alpha_2,
            lambda_1=lambda_1, lambda_2=lambda_2,
            compute_score=True
        )

    def fit(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None):
        """Fit Bayesian Ridge regression."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None,
                return_std: bool = False):
        """Predict using fitted Bayesian Ridge model."""
        if return_std:
            return self.model.predict(X, return_std=True)
        return self.model.predict(X)

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            coords_train: Optional[np.ndarray] = None,
                            coords_val: Optional[np.ndarray] = None,
                            metric: str = 'beta_nll',
                            verbose: bool = True) -> 'GlobalBayesianRidge':
        # grid of hyperparameters
        alpha_1_values = [1e-6, 1e-3, 1e0, 1e2]
        lambda_1_values = [1e-6, 1e-3, 1e0, 1e2]
        alpha_2_values = [1e-6, 1e-3, 1e0, 1e2]
        lambda_2_values = [1e-6, 1e-3, 1e0, 1e2]

        best_score = float('inf')
        best_params = None
        best_model = None
        n_configs = 0

        if verbose:
            print(f"Tuning Global Bayesian Ridge (metric: {metric})...")

        for alpha_1 in alpha_1_values:
            for alpha_2 in alpha_2_values:
                for lambda_1 in lambda_1_values:
                    for lambda_2 in lambda_2_values:
                        n_configs += 1
                        model = GlobalBayesianRidge(
                            alpha_1=alpha_1, alpha_2=alpha_2,
                            lambda_1=lambda_1, lambda_2=lambda_2
                        )
                        model.fit(X_train, y_train)
                        if metric == 'beta_nll':
                            score, _, _ = model.score(X_val, y_val, metric=metric)
                        else:
                            score, _ = model.score(X_val, y_val, metric=metric)

                        if score < best_score:
                            best_score = score
                            best_params = (alpha_1, alpha_2, lambda_1, lambda_2)
                            best_model = model

        if verbose:
            print(f"Best params: alpha_1={best_params[0]:.0e},\
                  alpha_2={best_params[1]:.0e},\
                  lambda_1={best_params[2]:.0e},\
                  lambda_2={best_params[3]:.0e}")
            print(f"Best {metric.upper()}: {best_score:.4f}\n")

        return best_model, best_params, n_configs



class GaussianProcess(BaselineModel):
    """
     Gaussian Process.
    """

    def __init__(self, kernel_type: str = 'rbf', length_scale: float = 1.0,
                 use_features: bool = True, n_restarts: int = 5):
       
        super().__init__(name="Kriging", supports_uncertainty=True)
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.use_features = use_features
        self.n_restarts = n_restarts

        # Build kernel
        kernel = self._build_kernel(kernel_type, length_scale)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=1e-2,  # Regularization for numerical stability
            normalize_y=True
        )

    def _build_kernel(self, kernel_type: str, length_scale: float):
        """Build GP kernel with narrower length scale bounds for better calibration."""
       
        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=length_scale,
                length_scale_bounds=(0.05, 10.0)  # Narrowed from (1e-2, 1e3)
            )
        elif kernel_type == 'matern':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=(0.05, 10.0),  # Narrowed from (1e-2, 1e3)
                nu=1.5
            )
        elif kernel_type == 'exponential':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=(0.05, 10.0),  # Narrowed from (1e-2, 1e3)
                nu=0.5
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

       
        kernel = kernel + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray):
        """
        Fit Kriging model.

        Args:
            X: Feature matrix
            y: Target values
            coords: Spatial coordinates (n_samples, 2)
        """
        if coords is None:
            raise ValueError("Kriging requires spatial coordinates")

        # Combine features with coordinates
        if self.use_features:
            X_combined = X  # contains coordinates info 
        else:
            # Ordinary kriging: use only coordinates
            X_combined = coords

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.model.fit(X_combined, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, coords: np.ndarray,
                return_std: bool = False):
        if coords is None:
            raise ValueError("Kriging requires spatial coordinates")

        if self.use_features:
            X_combined = X
        else:
            X_combined = coords

        if return_std:
            return self.model.predict(X_combined, return_std=True)
        return self.model.predict(X_combined)

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            coords_train: Optional[np.ndarray] = None,
                            coords_val: Optional[np.ndarray] = None,
                            metric: str = 'beta_nll',
                            verbose: bool = True,
                            use_features_options = False) -> 'GaussianProcess':
    
        kernel_types =['rbf', 'matern', 'exponential' ]
        length_scales = [1.0]

        best_score = float('inf')
        best_config = None
        best_model = None
        n_configs = 0

        if verbose:
            print(f"Tuning GaussianProcess (metric: {metric})...")
            print("  Note: This may take a while for large datasets...")

        for kernel_type in kernel_types:
            for length_scale in length_scales:
                n_configs += 1
                use_features = use_features_options
                try:
                    model = GaussianProcess(
                        kernel_type=kernel_type,
                        length_scale=length_scale,
                        use_features=use_features,
                        n_restarts=10
                    )
                    model.fit(X_train, y_train, coords_train)
                    if metric == 'beta_nll':
                        score, _, _ = model.score(X_val, y_val, coords_val, metric=metric)
                    else:
                        score, _ = model.score(X_val, y_val, coords_val, metric=metric)

                    if verbose:
                        print(f"  {kernel_type}, ls={length_scale} -> {metric.upper()}={score:.4f}")

                    if score < best_score:
                        best_score = score
                        best_config = (kernel_type, length_scale)
                        best_model = model
                except Exception as e:
                    if verbose:
                        print(f"  {kernel_type}, ls={length_scale} -> Failed: {e}")

        if verbose:
            print(f"Best config: kernel={best_config[0]}, length_scale={best_config[1]}")
            print(f"Best {metric.upper()}: {best_score:.4f}\n")

        return best_model, best_config, n_configs



class PyKriging(BaselineModel):
    """
    Universal Kriging.
    """

    def __init__(self, variogram_model: str = 'gaussian', nlags: int = 6):
        super().__init__(name="UK", supports_uncertainty=True)
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray):
        del X  # UK uses coordinates only; X kept for API consistency
        if coords is None:
            raise ValueError("PyKriging requires spatial coordinates")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            self.model = UniversalKriging(
                coords[:, 0],           # x (longitude)
                coords[:, 1],           # y (latitude)
                y,
                variogram_model=self.variogram_model,
                drift_terms=['regional_linear'],  # linear spatial trend
                nlags=self.nlags,
                enable_plotting=False,
                verbose=False,
            )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, coords: np.ndarray,
                return_std: bool = False):
        if coords is None:
            raise ValueError("PyKriging requires spatial coordinates")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            # Batch prediction faster than point-by-point loop
            z, ss = self.model.execute('points', coords[:, 0], coords[:, 1])

        predictions = np.asarray(z)
        stds = np.sqrt(np.maximum(np.asarray(ss), 0.0))
        if return_std:
            return predictions, stds
        return predictions

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            coords_train: Optional[np.ndarray] = None,
                            coords_val: Optional[np.ndarray] = None,
                            metric: str = 'beta_nll',
                             verbose: bool = True) -> 'PyKriging':
        variogram_models = ['spherical', 'exponential', 'gaussian', 'linear']
        nlags_options = [6] # default is 6, can also try [4, 8] but it will be very slow for large datasets, so we keep it fixed for now

        best_score = float('inf')
        best_config = None
        best_model = None
        n_configs = 0

        if verbose:
            print(f"Tuning Universal Kriging (metric: {metric})...")

        for variogram_model in variogram_models:
            for nlags in nlags_options:
                n_configs += 1
                try:
                    model = PyKriging(variogram_model=variogram_model, nlags=nlags)
                    model.fit(X_train, y_train, coords_train)

                    if metric == 'beta_nll':
                        score, _, _ = model.score(X_val, y_val, coords_val, metric=metric)
                    else:
                        score, _ = model.score(X_val, y_val, coords_val, metric=metric)

                    if verbose:
                        print(f"  {variogram_model}, nlags={nlags} -> {metric.upper()}={score:.4f}")

                    if score < best_score:
                        best_score = score
                        best_config = (variogram_model, nlags)
                        best_model = model
                except Exception as e:
                    if verbose:
                        print(f"  {variogram_model}, nlags={nlags} -> Failed: {e}")

        if best_model is None:
            raise RuntimeError("All PyKriging configurations failed during tuning.")

        if verbose:
            print(f"Best config: variogram={best_config[0]}, nlags={best_config[1]}")
            print(f"Best {metric.upper()}: {best_score:.4f}\n")

        return best_model, best_config, n_configs
    

class KNNRegression(BaselineModel):
    """
    K-Nearest Neighbors.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'distance'):
        
        super().__init__(name="KNN", supports_uncertainty=True)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='auto'
        )
        # Store training data for uncertainty estimation
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None):

        self.model.fit(X, y)
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        self.residual = y - self.model.predict(X)
        self.sigma_res = np.var(self.residual) + 1e-6
        return self

    def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None,
                return_std: bool = False):
        preds = self.model.predict(X)

        if return_std:
            # Compute std from variance of nearest neighbors
            _, indices = self.model.kneighbors(X)
            # get target values of nearest neighbors
            neighbor_values = self.y_train[indices]  # shape: (n_samples, n_neighbors)
            # compute std deviation across neighbors
            std = np.std(neighbor_values, axis=1)
            # avoid zero std
            local_std = np.maximum(std, 1e-6)
            # add residual 
            total_std = np.sqrt(local_std**2 + self.sigma_res)
            return preds, total_std

        return preds

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            coords_train: Optional[np.ndarray] = None,
                            coords_val: Optional[np.ndarray] = None,
                            metric: str = 'beta_nll',
                            verbose: bool = True) -> 'KNNRegression':
    
        n_neighbors_options = [3, 5, 10, 15, 20]
        weights_options = ['uniform', 'distance']

        best_score = float('inf')
        best_config = None
        best_model = None
        n_configs = 0

        if verbose:
            print(f"Tuning KNN (metric: {metric})...")

        for n_neighbors in n_neighbors_options:
            for weights in weights_options:
                n_configs += 1
                try:
                    model = KNNRegression(
                        n_neighbors=n_neighbors,
                        weights=weights
                    )
                    model.fit(X_train, y_train)

                    if metric == 'beta_nll':
                        score, _, _ = model.score(X_val, y_val, metric=metric)
                    else:
                        score, _ = model.score(X_val, y_val, coords_val, metric=metric)

                    if verbose:
                        print(f"  n={n_neighbors}, weights={weights} -> {metric.upper()}={score:.4f}")

                    if score < best_score:
                        best_score = score
                        best_config = (n_neighbors, weights)
                        best_model = model
                except Exception as e:
                    if verbose:
                        print(f"  n={n_neighbors}, weights={weights} -> Failed: {e}")

        if verbose:
            print(f"Best config: n_neighbors={best_config[0]}, weights={best_config[1]}")
            print(f"Best {metric.upper()}: {best_score:.4f}\n")

        return best_model, best_config, n_configs


class RandomForestRegression(BaselineModel):

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1, n_jobs: int = -1, random_state: int = 42):
        super().__init__(name="Random Forest", supports_uncertainty=True)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True     
        )

    def fit(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None):
        self.model.fit(X, y)
        self.is_fitted = True
        self.sigma_res = np.var(y - self.model.predict(X)) + 1e-6
        return self

    def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None,
                return_std: bool = False):
        preds = self.model.predict(X)
        if return_std:
            X_arr = np.asarray(X)
            tree_preds = np.array([tree.predict(X_arr) for tree in self.model.estimators_])
            local_std = np.maximum(np.std(tree_preds, axis=0), 1e-6)
            total_std = np.sqrt(local_std**2 + self.sigma_res)
            return preds, total_std
        return preds

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            coords_train: Optional[np.ndarray] = None,
                            coords_val: Optional[np.ndarray] = None,
                            metric: str = 'beta_nll',
                             verbose: bool = True):
        n_estimators_options  = [50, 100, 200]
        max_depth_options     = [None, 5, 10]
        min_samples_leaf_opts = [1, 5]

        best_score  = float('inf')
        best_config = None
        best_model  = None
        n_configs   = 0

        if verbose:
            print(f"Tuning Random Forest (metric: {metric})...")

        for n_est in n_estimators_options:
            for depth in max_depth_options:
                for leaf in min_samples_leaf_opts:
                    n_configs += 1
                    model = RandomForestRegression(
                        n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf
                    )
                    model.fit(X_train, y_train)

                    if metric == 'beta_nll':
                        score, _, _ = model.score(X_val, y_val, metric=metric)
                    else:
                        score, _ = model.score(X_val, y_val, metric=metric)

                    if verbose:
                        print(f"  n={n_est}, depth={depth}, leaf={leaf} -> {metric.upper()}={score:.4f}")

                    if score < best_score:
                        best_score  = score
                        best_config = (n_est, depth, leaf)
                        best_model  = model

        if verbose:
            print(f"Best: n_estimators={best_config[0]}, max_depth={best_config[1]}, "
                  f"min_samples_leaf={best_config[2]}")
            print(f"Best {metric.upper()}: {best_score:.4f}\n")

        return best_model, best_config, n_configs


def tune_and_evaluate_all_baselines(
    train_instances: DataFrame,
    val_instances: DataFrame,
    test_instances: DataFrame,
    target_col: str = 'label',
    models_to_run: List[str] = ['ridge', 'bayesian', 'gp', 'kriging', 'knn', 'rf'],
    verbose: bool = True,
    save_path: Optional[str] = None,
    metric: str = 'beta_nll',
    beta: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
   
    if beta is not None:
        global BETA_NLL_BETA
        BETA_NLL_BETA = beta

    # Extract features, targets, and coordinates
    feature_cols = []
    for c in train_instances.columns:
        if c not in ['label', 'geometry']:
            feature_cols.append(c)
    X_train = train_instances[feature_cols].values
    y_train = train_instances[target_col].values
    coords_train = extract_coordinates(train_instances)

    X_val = val_instances[feature_cols].values
    y_val = val_instances[target_col].values
    coords_val = extract_coordinates(val_instances)

    X_test = test_instances[feature_cols].values
    y_test = test_instances[target_col].values
    coords_test = extract_coordinates(test_instances)

    results = {}

    def _save():
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "baseline_results_partial.pkl"), "wb") as f:
                pickle.dump(results, f)

    # Global Bayesian 
    if 'bayesian' in models_to_run:
        if verbose:
            print("="*70)
            print("GLOBAL BAYESIAN RIDGE")
            print("="*70)

        t0 = time.time()
        bayesian, best_params, n_configs = GlobalBayesianRidge.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            coords_train, coords_val, verbose=verbose, metric=metric
        )

        test_rmse, pred = bayesian.score(X_test, y_test, metric='rmse')
        beta_nll_l, pred, std = bayesian.score(X_test, y_test, metric='beta_nll')
        elapsed = time.time() - t0

        results['bayesian'] = {
            'model': bayesian,
            'test_pred': pred,
            'test_std': std,
            'test_y': y_test,
            'test_rmse': test_rmse,
            'test_beta_nll': beta_nll_l,
            'best_params': best_params,
            'tuning_time_s': elapsed,
            'n_configs': n_configs,
            'time_per_config_s': elapsed / n_configs,
        }
        _save()

        if verbose:
            print(f"Test RMSE: {test_rmse:.4f}, Test Beta-NLL: {beta_nll_l:.4f}\n")


    # KNN
    if 'knn' in models_to_run:
        if verbose:
            print("="*70)
            print("K-NEAREST NEIGHBORS REGRESSION")
            print("="*70)

        t0 = time.time()
        knn, best_params, n_configs = KNNRegression.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            coords_train, coords_val, metric=metric, verbose=verbose
        )

        test_rmse, pred = knn.score(X_test, y_test, coords_test, metric='rmse')
        beta_nll_l, pred, std = knn.score(X_test, y_test, coords_test, metric='beta_nll')
        elapsed = time.time() - t0

        results['knn'] = {
            'model': knn,
            'test_pred': pred,
            'test_std': std,
            'test_y': y_test,
            'test_rmse': test_rmse,
            'test_beta_nll': beta_nll_l,
            'best_params': best_params,
            'tuning_time_s': elapsed,
            'n_configs': n_configs,
            'time_per_config_s': elapsed / n_configs,
        }
        _save()

        if verbose:
            print(f"Test RMSE: {test_rmse:.4f}, Test Beta-NLL: {beta_nll_l:.4f}\n")

    
    # Guassian process
    if 'gp' in models_to_run:
        if verbose:
            print("="*70)
            print(" GAUSSIAN PROCESS REGRESSION")
            print("="*70)

        t0 = time.time()
        gp, best_params, n_configs = GaussianProcess.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            coords_train, coords_val,
            metric=metric, verbose=verbose,
            use_features_options=True
        )

        test_rmse, pred = gp.score(X_test, y_test, coords_test, metric='rmse')
        beta_nll_l, pred, std = gp.score(X_test, y_test, coords_test, metric='beta_nll')
        elapsed = time.time() - t0

        results['gp'] = {
            'model': gp,
            'test_pred': pred,
            'test_std': std,
            'test_y': y_test,
            'test_rmse': test_rmse,
            'test_beta_nll': beta_nll_l,
            'best_params': best_params,
            'tuning_time_s': elapsed,
            'n_configs': n_configs,
            'time_per_config_s': elapsed / n_configs,
        }
        _save()
        if verbose:
            print(f"Test RMSE: {test_rmse:.4f}, Test Beta-NLL: {beta_nll_l:.4f}\n")

    # PyKriging 
    if 'kriging' in models_to_run:
        if verbose:
            print("="*70)
            print("PYKRIGING (PyKrige Library)")
            print("="*70)

        t0 = time.time()
        pykriging, best_params, n_configs = PyKriging.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            coords_train, coords_val, metric=metric, verbose=verbose
        )

        test_rmse, pred = pykriging.score(X_test, y_test, coords_test, metric='rmse')
        beta_nll_l, pred, std = pykriging.score(X_test, y_test, coords_test, metric='beta_nll')
        elapsed = time.time() - t0

        results['kriging'] = {
            'model': pykriging,
            'test_pred': pred,
            'test_std': std,
            'test_y': y_test,
            # 'temperature': temperature,
            'test_rmse': test_rmse,
            'test_beta_nll': beta_nll_l,
            'best_params': best_params,
            'tuning_time_s': elapsed,
            'n_configs': n_configs,
            'time_per_config_s': elapsed / n_configs,
        }
        _save()
        if verbose:
            print(f"Test RMSE: {test_rmse:.4f}, Test Beta-NLL: {beta_nll_l:.4f}\n")

    # Random Forest
    if 'rf' in models_to_run:
        if verbose:
            print("="*70)
            print("RANDOM FOREST REGRESSION")
            print("="*70)

        t0 = time.time()
        rf, best_params, n_configs = RandomForestRegression.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            coords_train, coords_val, metric=metric, verbose=verbose
        )

        test_rmse, pred   = rf.score(X_test, y_test, metric='rmse')
        beta_nll_l, pred, std = rf.score(X_test, y_test, coords_test, metric='beta_nll')
        elapsed = time.time() - t0

        results['rf'] = {
            'model':       rf,
            'test_pred':   pred,
            'test_std':    std,
            'test_y':      y_test,
            'test_rmse':   test_rmse,
            'test_beta_nll':   beta_nll_l,
            'best_params': best_params,
            'tuning_time_s': elapsed,
            'n_configs': n_configs,
            'time_per_config_s': elapsed / n_configs,
        }
        _save()

        if verbose:
            print(f"Test RMSE: {test_rmse:.4f}, Test Beta-NLL: {beta_nll_l:.4f}\n")
    return results


