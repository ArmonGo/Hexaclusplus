"""
Uncertainty-Aware Hexagonal Clustering for Spatial Regression

This module implements an adaptive clustering algorithm that:
1. Partitions space into H3 hexagons
2. Trains local Bayesian Ridge regression models in each hexagon
3. Iteratively merges hexagons using multi-objective optimization
4. Tracks and minimizes prediction uncertainty

Author: HexaClusV2 Team
Version: 2.0 (Uncertainty-aware)
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from collections import defaultdict
import copy
import os
import random
import warnings
import pickle
import pandas as pd
from polygon import SraiConstructor
from utils import gaussian_nll, combined_score
from baseline_models import KNNRegression, RandomForestRegression

# Supported kernel types
SUPPORTED_KERNELS = ['bayesian', 'knn', 'rf', 'gaussian']

    
def create_kernel_model(clustering: "Clustering") -> Any:
    """
    Create a regression model based on the kernel type.
    """
    kernel = clustering.kernel
    params = clustering.kernel_params

    if kernel == 'bayesian':
        return BayesianRidge(
            alpha_1=params.get('alpha_1', 1e-6),
            alpha_2=params.get('alpha_2', 1e-6),
            lambda_1=params.get('lambda_1', 1e-6),
            lambda_2=params.get('lambda_2', 1e-6),
            compute_score=True
        )
    elif kernel == 'knn':
        return KNNRegression(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'distance')
        )
    elif kernel == 'rf':
        return RandomForestRegression(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),     # shallow trees for local model 
            min_samples_leaf=params.get('min_samples_leaf', 3)
        )
    elif kernel == 'gaussian':
        k = (
            ConstantKernel(1.0, (1e-5, 1e3))
            * Matern(
                length_scale=params.get('length_scale', 1.0),
                length_scale_bounds=(1e-3, 100.0),  # wider for some dataset variability, but prevents overfitting on very small n
                nu=1.5
            )
            + WhiteKernel(
                noise_level=params.get('noise_level', 1e-2),
                noise_level_bounds=(1e-3, 1e1)  # floor at 1e-3 prevents interpolation on small local n
            )
        )
        return GaussianProcessRegressor(
            kernel=k,
            alpha=1e-6,          # numerical jitter only; noise captured by WhiteKernel
            n_restarts_optimizer=1,  # keep local fits fast
            normalize_y=True
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel}. Supported: {SUPPORTED_KERNELS}")


def train_polygon_model(
    clustering: "Clustering",
    polygon_idxs: List[int]
) -> Tuple[float, int, Any]:
    """
    Train a regression model over instances in specified polygons.
    """
    # Remove duplicates
    polygon_idxs = list(set(polygon_idxs))
    # Aggregate features and labels from all polygons
    instance_features = []
    instance_labels = []
    for idx in polygon_idxs:
        instances = clustering.get_instances_in_polygon(idx)
        instance_features.append(
            instances.drop(columns=["geometry", "label"]).values
        )
        instance_labels.extend(instances["label"].values)
    # Handle empty polygon case
    if not instance_features:
        return 0.0, 0, None

    X = np.vstack(instance_features)
    y = np.array(instance_labels)

    n_folds = getattr(clustering, 'n_cv_folds', 3)

    def _score(y_true, y_mean, y_std):
        if clustering.scoring_method == 'combined_score':
            return combined_score(y_true, y_mean, y_std, clustering.uncertainty_weight)
        elif clustering.scoring_method == 'nll':
            return gaussian_nll(y_true, y_mean, y_std)
        elif clustering.scoring_method == 'mse':
            return mean_squared_error(y_true, y_mean)
        else:
            raise KeyError(f"Scoring method '{clustering.scoring_method}' is not applicable!")

    # Cross-validate score to avoid in-sample bias when driving merge decisions.
    # Falls back to in-sample only when the polygon is too small for CV (shouldn't
    # happen in practice given min_samples_per_hexagon >= 30).
    if len(X) >= n_folds * 5:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            m = create_kernel_model(clustering)
            m.fit(X[train_idx], y[train_idx])
            y_m, y_s = m.predict(X[val_idx], return_std=True)
            fold_scores.append(_score(y[val_idx], y_m, y_s))
        score = float(np.mean(fold_scores))
    else:
        m_tmp = create_kernel_model(clustering)
        m_tmp.fit(X, y)
        y_m, y_s = m_tmp.predict(X, return_std=True)
        score = _score(y, y_m, y_s)

    # Refit on all data this model is stored and used for actual predictions.
    model = create_kernel_model(clustering)
    model.fit(X, y)

    return score, len(X), model



# ==============================================================================
# MAIN CLUSTERING CLASS
# ==============================================================================

class Clustering:
    """
    Uncertainty-aware spatial clustering with iterative hexagon merging.

    The algorithm:
    * Initializes H3 hexagons over the study area
    * Trains local regression models in each hexagon
    * Iteratively merges adjacent hexagons to improve performance
    * Uses nll optimization (Balance between accuracy and uncertainty) to drive merges
    * Employs simulated anneling for better exploration
    * Stops when validation performance plateaus

    """

    def __init__(
        self,
        instances: GeoDataFrame, # Training GeoDataFrame with 'geometry' and 'label' columns
        val_instances: GeoDataFrame,
        test_instances: Optional[GeoDataFrame] = None,
        save_path: str = './algo/',
        measurements: Optional[List[GeoDataFrame]] = None, # Additional spatial measurements to aggregate, not required for this version
        selected_area: str = None, # e.g., "London, UK"
        resolution: int = 6, # H3 hexagon resolution, higher = smaller hexagons. Different areas use different resolutions to balance granularity and computational cost
        kernel: str = 'bayesian',
        kernel_params: Optional[dict] = None,  # Hyperparamters for different kernels 
        uncertainty_weight: float = 0.1, # Discarded in this version, use nll directly as scoring method instead of combined score
        min_samples_per_hexagon: int = 20, # 30 in experiemnts to keep the statiscal reliability of uncertainty estimation
        scoring_method: str = 'nll',
        min_polygons: int = 2, 
        merge_threshold: float = 0.0,
        n_cv_folds: int = 3
    ):
        
        # Validate kernel
        if kernel not in SUPPORTED_KERNELS:
            raise ValueError(f"Unknown kernel '{kernel}'. Supported: {SUPPORTED_KERNELS}")

        # Store data
        self.instances = instances
        self.val_instances = val_instances
        self.test_instances = test_instances
        self.measurements = measurements if measurements is not None else []
        self.save_path = save_path
        # Kernel configuration
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.uncertainty_weight = uncertainty_weight
        # Minimum sample size constraint
        self.min_samples_per_hexagon = min_samples_per_hexagon
        # Merge control parameters
        self.min_polygons = min_polygons
        self.merge_threshold = merge_threshold
        self.n_cv_folds = n_cv_folds

        # Initialize data structures for clustering
        self.polygons: List[Polygon] = []  # Hexagon geometries
        self.polygon_states: List[int] = []  # 1=active, 0=merged
        self.touching_pairs: List[Tuple[int, int]] = []  # Adjacent pairs
        self.models: List[Any] = []  # Trained regression models
        self.scoring_method: str  = scoring_method
        self.score : List[float] = []
        self.score_diff: List[float] = []  # score improvement if merged
        self.length: List[int] = []  # Number of instances per pair
        self.instance_assignments: List[List[int]] = []  # Train instances per polygon
        self.val_instance_assignments: List[List[int]] = []  # Val instances per polygon
        self.polygon_neighbors = defaultdict(set)  
        self.history: List[List[int]] = []  # Active polygons at each step
        self.resolution = resolution

        # Initialize hexagon constructor, every run may return different embeddings 
        self._constructor = SraiConstructor(
            selected_area=selected_area,
            resolution=resolution,
            encoder_sizes=[10, 5],
            add_offset_features=True,
            min_samples_per_hexagon=min_samples_per_hexagon
        )

        self.geo_feats = []  # Spatial features
        self.boundary = None  # Study area boundary

    def initialize(self, polygons: List[Polygon], #  List of hexagon geometries
                   feats: Optional[np.ndarray] = None) -> None:  #  Hex2Vec embeddings
        self.polygons = polygons
        self.boundary = unary_union(self.polygons)

        # Aggregate spatial features
        if feats is not None:
            self.geo_feats.append(feats)

        if len(self.measurements) > 0:
            gdf_measurements = self.aggregate_features_by_polygon(
                self.measurements, self.polygons
            )
            self.geo_feats.append(
                np.array(gdf_measurements.drop(columns=['geometry']))
            )

        # Concatenate and normalize features
        self.geo_feats = np.concatenate(self.geo_feats, axis=1)
        scaler = StandardScaler()
        self.geo_feats = scaler.fit_transform(self.geo_feats)

        # Store as GeoDataFrame for spatial joins
        self.geo_feats = GeoDataFrame(
            geometry=self.polygons,
            data={i: list(self.geo_feats[:, i])
                  for i in range(self.geo_feats.shape[1])}
        )

        # Add spatial features to instances
        self.instances = self.append_geo_features(self.instances).reset_index(drop=True)
        self.val_instances = self.filter_instances(self.val_instances)
        self.val_instances = self.append_geo_features(self.val_instances).reset_index(drop=True)

        if self.test_instances is not None:
            self.test_instances = self.filter_instances(self.test_instances)
            self.test_instances = self.append_geo_features(self.test_instances)

        # Clear and reinitialize data structures
        self.clear_memory()

        # Assign instances to polygons
        self.instance_assignments = self.assign_instance_dict(
            copy.deepcopy(self.instances),
            copy.deepcopy(self.polygons)
        )
        self.val_instance_assignments = self.assign_instance_dict(
            copy.deepcopy(self.val_instances),
            copy.deepcopy(self.polygons)
        )

        # Find touching hexagon pairs
        touching_gdf = gpd.sjoin(
            GeoDataFrame(geometry=self.polygons),
            GeoDataFrame(geometry=self.polygons),
            predicate="touches"
        )
        touching_pairs = list(set(
            tuple(sorted((i, j)))
            for i, j in zip(touching_gdf.index, touching_gdf.index_right)
        ))

        # Train initial models for each hexagon (self-pairs)
        for p_ix in range(len(self.polygons)):
            score, length, model = train_polygon_model(self, [p_ix])
            self.models.append(model)
            self.polygon_states.append(1)  # Active
            self.touching_pairs.append((p_ix, p_ix))
            self.score.append(score)
            self.length.append(length)

        # Train models for touching pairs
        for pair in touching_pairs:
            self.touching_pairs.append(pair)
            score, length, _ = train_polygon_model(self, list(pair))
            self.score.append(score)
            self.length.append(length)

        # Calculate score improvement for each pair
        for pair in self.touching_pairs:
            self.score_diff.append(self.get_score_diff(pair))

        # Build adjacency graph
        for key, value in self.touching_pairs:
            self.polygon_neighbors[key].add(value)
            self.polygon_neighbors[value].add(key)
        self.polygon_neighbors = {k: sorted(v) for k, v in self.polygon_neighbors.items()}

    def clear_memory(self):
        """Clear all data structures (used during reinitialization)."""
        self.touching_pairs.clear()
        self.length.clear()
        self.score.clear()
        self.models.clear()
        self.instance_assignments.clear()
        self.val_instance_assignments.clear()
        self.score_diff.clear()
        self.history.clear()
        self.polygon_neighbors = defaultdict(set)


# Feature aggregation and spatial join methods
    def aggregate_features_by_polygon(
        self,
        measurements: List[GeoDataFrame],
        polygons: List[Polygon]
    ) -> GeoDataFrame:
        """
        Aggregate features from multiple GeoDataFrames into hexagons
        """
        gdf_bins = GeoDataFrame(
            geometry=polygons,
            data={'polygon_ix': list(range(len(polygons)))}
        )

        # Collect all feature columns
        all_feature_cols = set()
        for gdf in measurements:
            all_feature_cols.update(gdf.columns.difference(['geometry']))

        # Standardize columns across all GeoDataFrames
        standardized_gdfs = []
        for gdf in measurements:
            for col in all_feature_cols:
                if col not in gdf.columns:
                    gdf[col] = None
            standardized_gdfs.append(gdf)

        # Combine and spatially join with polygons
        gdf_combined = gpd.GeoDataFrame(
            pd.concat(standardized_gdfs, ignore_index=True),
            crs=measurements[0].crs
        )
        gdf_joined = gpd.sjoin(
            gdf_combined, gdf_bins,
            predicate='intersects', how='left'
        )

        # Aggregate by taking mean within each polygon
        feature_cols = list(all_feature_cols)
        gdf_aggregated = gdf_joined.groupby(gdf_bins.index)[feature_cols].mean()

        gdf_final = gdf_bins.copy()
        gdf_final = gdf_final.merge(
            gdf_aggregated,
            left_index=True,
            right_index=True,
            how='left'
        )
        gdf_final = gdf_final.drop(columns=['polygon_ix'])

        return gdf_final

    def filter_instances(self, instances: GeoDataFrame) -> GeoDataFrame:
        """
        Filter instances to only those within the study area boundary.
        """
        inside_mask = instances['geometry'].apply(lambda x: self.boundary.contains(x))
        num_outside = len(inside_mask) - sum(inside_mask)

        if num_outside > 0:
            warnings.warn(
                f"{num_outside} instances fall outside the boundary and will be removed"
            )

        return instances[inside_mask]

    def append_geo_features(self, instances: GeoDataFrame) -> GeoDataFrame:
        """
        Add spatial features to instances via spatial join with hexagons.
        Also adds node-level offset features.
        """
        # Join with hexagon features
        instances = gpd.sjoin(
            instances, self.geo_feats,
            predicate='intersects', how='left'
        )
        instances = instances.drop(
            columns=[col for col in instances.columns if 'index' in str(col)]
        )

        # Add offset features (position within hexagon)
        if self._constructor.add_offset_features:
            instances = self._constructor.compute_offset_features(
                instances, self.geo_feats
            )

        return instances

    def assign_instance_dict(
        self,
        instances: GeoDataFrame,
        polygons: List[Polygon]
    ) -> List[List[int]]:
        """
        Assign each instance to its containing polygon.
        """
        p_space = GeoDataFrame(geometry=polygons)
        joined = gpd.sjoin(p_space, instances, how="left", predicate="contains")

        instance_assignments = (
            joined.groupby(joined.index)
            .apply(lambda x: x.index_right.dropna().astype(int).tolist())
            .reindex(range(len(p_space)), fill_value=[])
            .tolist()
        )

        return instance_assignments

    
    # Methods for selecting which polygons to merge
    def get_valid_merge_pairs(self) -> Tuple[List, List]:
        """
        Get all valid merge pairs (both polygons active, not self-pairs).
        """
        valid_pairs = []
        valid_indices = []

        for i, pair in enumerate(self.touching_pairs):
            if pair[0] != pair[1]:  # Not self-pair
                if (self.polygon_states[pair[0]] == 1 and
                    self.polygon_states[pair[1]] == 1):  # Both active
                    valid_pairs.append(pair)
                    valid_indices.append(i)

        return valid_pairs, valid_indices
    
    def get_score_diff(self, pair: Tuple[int, int]) -> float:
        """
        Calculate score improvement if this pair is merged.
        """
        if pair[0] == pair[1]:
            return -float('inf')  # Cannot merge with itself

        ix_combined = self.touching_pairs.index(pair)
        ix_i = self.touching_pairs.index((pair[0], pair[0]))
        ix_j = self.touching_pairs.index((pair[1], pair[1]))

        # Weighted average score of separate polygons minus merged score
        score_improvement = (
            (self.length[ix_i] * self.score[ix_i] +
             self.length[ix_j] * self.score[ix_j]) / self.length[ix_combined]
            - self.score[ix_combined]
        )
        return score_improvement

    def get_merge_polygon_pairs(self) -> Tuple[Tuple[int, int], int]:
        """
        Select pair with maximum score improvement.
        """
        ix = np.argmax(self.score_diff).item()
        return self.touching_pairs[ix], ix

    # Merging
    def merge_polygons(
        self
    ) -> bool:
        """
        Merge the best polygon pair.
        """
        # Save current state to history
        _, active_indices = self.get_active_polygons()
        self.history.append(active_indices)
        # Check minimum polygon constraint
        if len(active_indices) <= self.min_polygons:
            print(f"  Reached minimum polygon count ({self.min_polygons}), stopping merges.")
            return False

        # Select best pair
        merge_pair, pair_idx = self.get_merge_polygon_pairs()
        if merge_pair[0] == merge_pair[1]:
            return False  # No valid merge

        # Check merge threshold constraint
        best_improvement = self.score_diff[pair_idx]
        if best_improvement < self.merge_threshold:
            print(f"  Best improvement ({best_improvement:.4f}) below threshold ({self.merge_threshold}), stopping.")
            return False
        return self._execute_merge(merge_pair)

    def merge_polygons_forced(self, merge_pair: Tuple[int, int]) -> bool:
        """
        Forcing merging, for the cases of empty hexagons or hexagons with very few samples. 
        """
        _, active_indices = self.get_active_polygons()
        self.history.append(active_indices)
        if merge_pair[0] == merge_pair[1]:
            return False
        return self._execute_merge(merge_pair)

    def _execute_merge(self, merge_pair: Tuple[int, int]) -> bool:
        # Create new merged polygon
        new_polygon_index = len(self.polygons)
        self.polygons.append(
            unary_union([self.polygons[merge_pair[0]],
                        self.polygons[merge_pair[1]]])
        )

        # Merge instance assignments
        self.instance_assignments.append(
            self.instance_assignments[merge_pair[0]] +
            self.instance_assignments[merge_pair[1]]
        )
        self.val_instance_assignments.append(
            self.val_instance_assignments[merge_pair[0]] +
            self.val_instance_assignments[merge_pair[1]]
        )

        # Mark old polygons as inactive
        self.polygon_states[merge_pair[0]] = 0
        self.polygon_states[merge_pair[1]] = 0
        self.polygon_states.append(1)  # New polygon is active

        # Update neighbor list
        new_neighbors = sorted(list(
            set(self.polygon_neighbors[merge_pair[0]] +
                self.polygon_neighbors[merge_pair[1]])
            - {merge_pair[0], merge_pair[1]}
            | {new_polygon_index}
        ))

        # Train model for new merged polygon
        self.touching_pairs.append((new_polygon_index, new_polygon_index))
        score, length, model = train_polygon_model(
            self, [new_polygon_index, new_polygon_index]
        )
        self.score.append(score)
        self.length.append(length)
        self.models.append(model)
        # Self-pairs cannot be merged, use -inf as placeholder
        self.score_diff.append(-float('inf'))
        self.polygon_neighbors[new_polygon_index] = new_neighbors

        # Train models for new pairs with neighbors
        for neighbor in new_neighbors[:-1]:  # Exclude self
            self.touching_pairs.append((neighbor, new_polygon_index))
            score, length, model = train_polygon_model(
                self, [neighbor, new_polygon_index]
            )
            self.score.append(score)
            self.length.append(length)
            self.score_diff.append(self.get_score_diff((neighbor, new_polygon_index)))
            self.polygon_neighbors[neighbor].append(new_polygon_index)

        # Remove old polygon data
        self.drop_old_polygons(merge_pair[0])
        self.drop_old_polygons(merge_pair[1])

        return True

    def drop_old_polygons(self, old_idx: int):
        """
        Remove all data for a merged (inactive) polygon.
        """
        neighbors = copy.deepcopy(self.polygon_neighbors[old_idx])

        for neighbor in neighbors:
            pair = tuple(sorted([neighbor, old_idx]))
            r_ix = self.touching_pairs.index(pair)
            self.touching_pairs.pop(r_ix)
            self.score.pop(r_ix)
            self.score_diff.pop(r_ix)
            self.length.pop(r_ix)
            self.polygon_neighbors[neighbor].remove(old_idx)

        del self.polygon_neighbors[old_idx]

    # Optimization step with simulated annealing
    def simulated_annealing_step(
        self,
        temperature: float
    ) -> bool:
        """
        Perform one simulated annealing step.
        With probability 30%, explores a random merge.
        Otherwise, performs greedy best merge.
        """
        # Check minimum polygon constraint first
        _, active_indices = self.get_active_polygons()
        if len(active_indices) <= self.min_polygons:
            print(f"  Reached minimum polygon count ({self.min_polygons}), stopping merges.")
            return False

        _, valid_indices = self.get_valid_merge_pairs()  # exclude self pair
        if not valid_indices:
            return False

        # Calculate best score
        current_scores = [self.score_diff[i] for i in valid_indices]
        best_score = max(current_scores)

        # Check merge threshold
        if best_score < self.merge_threshold:
            print(f"  Best improvement ({best_score:.4f}) below threshold ({self.merge_threshold}), stopping.")
            return False

        # Exploration: try random merge with some probability
        if random.random() < 0.3:  # 30% exploration rate
            random_idx = random.choice(valid_indices)
            random_score = self.score_diff[random_idx]

            # Accept if better, or probabilistically if worse
            delta = random_score - best_score
            if delta > 0 or random.random() < np.exp(delta / (temperature + 1e-10)):
                merge_pair = self.touching_pairs[random_idx]
                return self.merge_polygons_forced(merge_pair)

        # Exploitation: use best merge (greedy)
        return self.merge_polygons()

    # construct clustering and main merge loop 
    def construct_clustering(
        self,
        max_iter: int = 10000,
        patience: int = 100,
        restart: bool = True,
        use_simulated_annealing: bool = False,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Iteratively merge hexagons.
        """
        val_score_list = []
        test_score_list = []
        score_list = []  # Track the actual scoring metric used

        # Initialize hexagons
        if restart:
            if not self._constructor:
                raise ValueError("No constructor initialized")

            # Check for cached polygons and features (no need to constrct the hex2vec every time)
            cache_path = os.path.join(self.save_path, f"spatial_cache_res{self.resolution}.pkl")

            if os.path.exists(cache_path):
                print(f"Loading cached polygons and features (resolution={self.resolution})...")
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                polygons, features = cache["polygons"], cache["features"]
                print(f"  Loaded {len(polygons)} polygons")
            else:
                print(f"Constructing polygons and features (resolution={self.resolution}, will be cached)...")
                polygons, features = self._constructor.construct(self.instances, min_samples = self.min_samples_per_hexagon)
                os.makedirs(self.save_path, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({"polygons": list(polygons), "features": features}, f)
                print(f"  Cached to {cache_path}")

            self.initialize(polygons, features)
            self.best_score = float("inf")  # Single best score tracker

            # Report statistics after initialization
            _, active_indices = self.get_active_polygons()
            sample_counts = [len(self.instance_assignments[i]) for i in active_indices]
            print(f"Initialized with {len(active_indices)} hexagons")
            print(f"  Mean samples per hexagon: {np.mean(sample_counts):.1f}")
            print(f"  Min samples: {np.min(sample_counts)}, Max samples: {np.max(sample_counts)}")

        # Start merging
        merges = 0
        tol = 0  # Patience counter
        temperature = initial_temp

        print(f'Merging begins... (scoring method: {self.scoring_method})')

        while True:
            # Execute merge using consistent scoring method
            try:
                if use_simulated_annealing:
                    # SA uses multiobjective flag for compatibility
                    merge_success = self.simulated_annealing_step(
                        temperature
                    )
                    temperature *= cooling_rate

                    if temperature < min_temp:
                        print('Temperature reached minimum, stopping.')
                        break
                else:
                    # Select merge based on scoring method
                    merge_success = self.merge_polygons()
                if not merge_success:
                    print('No available polygons to merge.')
                    break
            except ValueError as e:
                print(f'Error during merge: {e}')
                break

            # Evaluate on validation set
            merges += 1
            val_score, _, _, val_uncertainty = self.validate()
            val_score_list.append(val_score)

            # Evaluate on test set
            if self.test_instances is not None:
                test_score, _, _, _ = self.predict(instances=self.test_instances)
                test_score_list.append(test_score)

            current_score = val_score
            score_list.append(current_score)

            # Check if this is the best model so far
            is_best = current_score < self.best_score

            # Save best model
            if is_best:
                self.best_score = current_score
                self.save_best_instance(
                    best_score=val_score,
                    best_uncertainty=np.mean(val_uncertainty),
                    score_name=self.scoring_method
                )
                tol = 0  # Reset patience
            else:
                tol += 1  # Increment patience

            # Stopping criteria
            if max_iter > 0 and merges >= max_iter:
                print(f'Reached maximum iterations ({max_iter}).')
                break
            if tol >= patience:
                print(f'Patience exhausted ({patience} iterations without improvement).')
                break

        print(f'Clustering complete. Total merges: {merges}')
        print(f'Best {self.scoring_method}: {self.best_score:.4f}')

        return val_score_list, test_score_list, score_list

    # Prediction
    def get_instances_in_polygon(
        self,
        polygon_idx: int,
        dict_instance: Optional[List] = None,
        instances_used: Optional[GeoDataFrame] = None
    ) -> GeoDataFrame:
        """
        Get all instances within a specified polygon.
        In test set, if the instances do not fall into the obeservation boundary, we will simply remove it from the evaluation.
        """
        if dict_instance is None:
            dict_instance = self.instance_assignments
            instances_used = self.instances

        if polygon_idx >= len(self.polygons) or self.polygons[polygon_idx] is None:
            raise ValueError(f"No polygon with index {polygon_idx}")

        return instances_used.iloc[list(dict_instance[polygon_idx])]

    def get_within_polygons_index(self, instance: GeoDataFrame) -> int:
        """
        Find which active polygon contains an instance.
        """
        poi = instance['geometry']
        active_polys, active_ixs = self.get_active_polygons()

        loc = [poly.contains(poi) for poly in active_polys]
        model_ix = [active_ixs[i] for i, contained in enumerate(loc) if contained]

        if len(model_ix) == 0:
            raise NotImplementedError('Instance not in any active polygon')

        if len(model_ix) > 1:
            raise NotImplementedError(
                f'Instance in {len(model_ix)} polygons (should be 1)'
            )

        return model_ix[0]

    def validate(self) -> Tuple[float, np.ndarray, List, float]:
        """
        Evaluate model on validation set.
        """
        _, active_ix = self.get_active_polygons()

        models = []
        X_list = []
        y_list = []

        # Collect data from each active polygon
        for idx in active_ix:
            if len(self.val_instance_assignments[idx]) > 0:
                models.append(self.models[idx])
                instances = self.get_instances_in_polygon(
                    idx,
                    self.val_instance_assignments,
                    self.val_instances
                )
                X_list.append(
                    np.array(instances.drop(columns=["geometry", "label"]))
                )
                y_list.extend(list(instances['label']))

        # Get predictions and uncertainties
        preds_list = []
        uncertainty_list = []

        for X, model in zip(X_list, models):
            # All supported kernels support return_std
            pred, pred_std = model.predict(X, return_std=True)
            preds_list.append(pred)
            uncertainty_list.extend(pred_std)

        # Flatten predictions
        y_mean = np.array([item for sublist in preds_list for item in sublist])
        y = np.array(y_list)
        y_std = np.array(uncertainty_list)

        if self.scoring_method == 'combined_score':
            score = combined_score(y, y_mean, y_std, self.uncertainty_weight)
        elif self.scoring_method == 'nll':
            score = gaussian_nll(y, y_mean, y_std)
        elif self.scoring_method == 'mse':
            score = mean_squared_error(y, y_mean)
        else:
            raise KeyError("scoring method is not applicable!")
        return score, y_mean, self.val_instance_assignments, y_std

    def predict(
        self,
        instances: GeoDataFrame
    ) -> Tuple[float, List[float], GeoDataFrame, List[float]]:
        # Filter and add spatial features
        instances = self.filter_instances(instances)
        instances = self.append_geo_features(instances)

        preds = []
        uncertainties = []

        # Predict each instance
        for i in range(len(instances)):
            # Find containing polygon
            polygon_idx = self.get_within_polygons_index(instances.iloc[i]) # Filter instances 
            # Extract features
            X = instances.drop(columns=["geometry", "label"]).iloc[i:i+1].values
            # Get prediction and uncertainty
            model = self.models[polygon_idx]
            # All supported kernels support return_std
            pred, pred_std = model.predict(X, return_std=True)
            preds.append(pred.item())
            uncertainties.append(pred_std.item())

        # Calculate metrics
        y = instances['label'].values
        preds_arr = np.array(preds)
        uncertainties_arr = np.array(uncertainties)

        if self.scoring_method == 'combined_score':
            score = combined_score(y, preds_arr, uncertainties_arr, self.uncertainty_weight)
        elif self.scoring_method == 'nll':
            score = gaussian_nll(y, preds_arr, uncertainties_arr)
        elif self.scoring_method == 'mse':
            score = mean_squared_error(y, preds_arr)
        else:
            raise KeyError("scoring method is not applicable!")

        return score, preds, instances, uncertainties

    # Help functions 
    def get_active_polygons(self) -> Tuple[List[Polygon], List[int]]:
        active_indices = [
            i for i in range(len(self.polygon_states))
            if self.polygon_states[i] == 1
        ]
        active_polygons = [self.polygons[i] for i in active_indices]

        return active_polygons, active_indices

    def save_best_instance(
        self,
        best_score: Optional[float] = None,
        best_uncertainty: Optional[float] = None,
        score_name: str = "Score"
    ):

        with open(self.save_path + "best_model.pkl", "wb") as f:
            pickle.dump(self, f)

        if best_score is not None:
            print(
                f"Best model saved - {score_name}: {best_score:.4f} "
                f"(Uncertainty: {best_uncertainty:.4f})"
            )
        else:
            print(f"Best model saved - {score_name}: {best_score:.4f}")

    @staticmethod
    def load_best_instance(save_path: str) -> "Clustering":
        with open(save_path + "best_model.pkl", "rb") as f:
            return pickle.load(f)
