
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import defaultdict
import copy
import os
import random
import warnings
import pickle
import pandas as pd
from polygon import SraiConstructor
from utils import combined_score, gaussian_beta_nll
from baseline_models import KNNRegression, RandomForestRegression

# Supported kernel types
SUPPORTED_KERNELS = ['bayesian', 'knn', 'rf', 'gaussian']

    
def _adaptive_n_estimators(base: int, cap: int, n_samples: Optional[int],
                           n0: Optional[int], mode: Optional[str]) -> int:
    if mode is None or n_samples is None or n0 is None or n0 <= 0:
        return base
    ratio = max(n_samples / n0, 1.0) 
    if mode == 'sqrt':
        scaled = base * np.sqrt(ratio)
    elif mode == 'log':
        scaled = base * (1.0 + np.log2(ratio))
    else:
        raise ValueError(f"Unknown trees_growth mode: {mode}. Use 'sqrt', 'log', or None.")
    return int(np.clip(round(scaled), base, cap))


def create_kernel_model(clustering: "Clustering", n_samples: Optional[int] = None) -> Any:
    kernel = clustering.kernel
    params = clustering.kernel_params

    if kernel == 'bayesian':
        alpha = params.get('alpha', 1e-6)
        lam = params.get('lambda', 1e-4)
        return BayesianRidge(
            alpha_1=alpha, alpha_2=alpha,
            lambda_1=lam, lambda_2=lam,
            max_iter=params.get('max_iter', 300),
            tol=params.get('tol', 1e-3),
            # compute_score=False
        )

    elif kernel == 'knn':
        return KNNRegression(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'distance')
        )
    elif kernel == 'rf':
        # Tree count scales with region size: small hexagons stay cheap, large merged
        # regions get a more stable ensemble. 
        n_trees = _adaptive_n_estimators(
            base=params.get('n_estimators', 100),
            cap=params.get('max_estimators', 300),
            n_samples=n_samples,
            n0=getattr(clustering, 'min_samples_per_hexagon', None),
            mode=params.get('trees_growth', None),
        )
        return RandomForestRegression(
            n_estimators=n_trees,
            max_depth=params.get('max_depth', 3),     # shallow 
            min_samples_leaf=params.get('min_samples_leaf', 3),
            n_jobs=-1,   
            random_state=getattr(clustering, 'random_seed', 42),
        )
    elif kernel == 'gaussian':
        k = (
            ConstantKernel(1.0, (0.1, 10.0))  
            * Matern(
                length_scale=params.get('length_scale') or 1.0,
                length_scale_bounds=(1e-2, 10.0),
                nu=1.5
            )
            + WhiteKernel(
                noise_level=params.get('noise_level') or 1e-2,
                noise_level_bounds=(1e-5, 1e1)
            )
        )
        return GaussianProcessRegressor(
            kernel=k,
            alpha=1e-4,              
            n_restarts_optimizer=0,  
            normalize_y=True
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel}. Supported: {SUPPORTED_KERNELS}")

def train_polygon_model(
    clustering: "Clustering",
    polygon_idxs: List[int]
) -> Tuple[float, int, Any]:
   
    # Remove duplicates
    polygon_idxs = list(set(polygon_idxs))

    # Aggregate features and labels from the region's own polygons
    instance_features = []
    instance_labels = []

    for idx in polygon_idxs:
        instances = clustering.get_instances_in_polygon(idx)
        instance_features.append(
            instances.drop(columns=["geometry", "label"]).values
        )
        instance_labels.extend(instances["label"].values)

    # handle empty polygon case
    if not instance_features:
        return 0.0, 0, None

    X = np.vstack(instance_features)
    y = np.array(instance_labels)

    # halo augmentation (optional): not use at the end 
    halo_buffer = getattr(clustering, 'halo_buffer', 0.0) or 0.0
    if halo_buffer > 0:
        halo = clustering.get_halo_instances(polygon_idxs)
        if len(halo) > 0:
            X_halo = halo.drop(columns=["geometry", "label"]).values
            y_halo = halo["label"].values
        else:
            X_halo, y_halo = np.empty((0, X.shape[1])), np.empty((0,))
    else:
        X_halo, y_halo = np.empty((0, X.shape[1])), np.empty((0,))

    n_fit = len(X) + len(X_halo)   # total training size (drives RF tree scaling)
    n_folds = getattr(clustering, 'n_cv_folds', 3)

    def _score(y_true, y_mean, y_std):
        if clustering.scoring_method == 'combined_score':
            return combined_score(y_true, y_mean, y_std, clustering.uncertainty_weight)
        elif clustering.scoring_method == 'beta_nll':
            return gaussian_beta_nll(y_true, y_mean, y_std, beta=clustering.beta)
        elif clustering.scoring_method == 'mse':
            return mean_squared_error(y_true, y_mean)
        else:
            raise KeyError(f"Scoring method '{clustering.scoring_method}' is not applicable!")


    if len(X) >= n_folds * 5:
        kf = KFold(n_splits=n_folds, shuffle=True,
                   random_state=getattr(clustering, 'random_seed', 42))
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr = np.vstack([X[train_idx], X_halo])
            y_tr = np.concatenate([y[train_idx], y_halo])
            m = create_kernel_model(clustering, n_samples=n_fit)
            m.fit(X_tr, y_tr)
            y_m, y_s = m.predict(X[val_idx], return_std=True)
            score = _score(y[val_idx], y_m, y_s)
            fold_scores.append(score)
        score = float(np.mean(fold_scores))
    else:
        X_tr = np.vstack([X, X_halo])
        y_tr = np.concatenate([y, y_halo])
        m_tmp = create_kernel_model(clustering, n_samples=n_fit)
        m_tmp.fit(X_tr, y_tr)
        y_m, y_s = m_tmp.predict(X, return_std=True)
        score = _score(y, y_m, y_s)

    # Refit on inside + halo — this model is stored and used for actual predictions.
    X_tr = np.vstack([X, X_halo])
    y_tr = np.concatenate([y, y_halo])
    model = create_kernel_model(clustering, n_samples=n_fit)
    model.fit(X_tr, y_tr)

    return score, len(X), model


class Clustering:

    def __init__(
        self,
        instances: GeoDataFrame,
        val_instances: GeoDataFrame,
        test_instances: Optional[GeoDataFrame] = None,
        save_path: str = './algo/',
        measurements: Optional[List[GeoDataFrame]] = None,
        selected_area: str = None,
        resolution: int = 6,
        kernel: str = 'bayesian',
        kernel_params: Optional[dict] = None,
        uncertainty_weight: float = 0.1,
        min_samples_per_hexagon: int = 20,
        scoring_method: str = 'beta_nll',
        beta: float = 0.5,
        min_polygons: int = 2,
        merge_threshold: float = 0.0,
        halo_buffer: float = 0.0,
        random_seed: int = 42,
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

        # kernel configuration
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.uncertainty_weight = uncertainty_weight
        self.min_samples_per_hexagon = min_samples_per_hexagon

        # Merge control
        self.min_polygons = min_polygons
        self.merge_threshold = merge_threshold
        self.halo_buffer = halo_buffer   # >0 enables halo training, =0 skips halo training
        self.random_seed = random_seed   # seeds SA exploration, cv folds, and the rf kernel
        self.n_cv_folds = n_cv_folds

        # Initialize data structures for clustering
        self.polygons: List[Polygon] = [] 
        self.polygon_states: List[int] = []  # 1=active, 0=merged
        self.touching_pairs: List[Tuple[int, int]] = []  
        self.models: List[Any] = []  
        self.scoring_method: str  = scoring_method
        self.beta = beta   # only used when scoring_method == 'beta_nll'
        self.score : List[float] = []
        self.score_diff: List[float] = []  # score improvement if merged
        self.length: List[int] = []  # nr. of instances per pair
        self.instance_assignments: List[List[int]] = []  # training instances per polygon
        self.val_instance_assignments: List[List[int]] = []  # validation instances per polygon
        self.polygon_neighbors = defaultdict(set)  # adj graph
        self.history: List[List[int]] = []  # active polygons at each step

        # Store resolution for cache naming
        self.resolution = resolution

        # Initialize hexagon constructor
        self._constructor = SraiConstructor(
            selected_area=selected_area,
            resolution=resolution,
            encoder_sizes=[10, 5],
            add_offset_features=True,
            min_samples_per_hexagon=min_samples_per_hexagon,
            random_seed=random_seed
        )

        self.geo_feats = []  # Spatial features
        self.boundary = None  # Study area boundary

    def initialize(self, polygons: List[Polygon], feats: Optional[np.ndarray] = None) -> None:
        
        self.polygons = polygons
        self.boundary = unary_union(self.polygons)

        if feats is not None:
            self.geo_feats.append(feats)

        if len(self.measurements) > 0:
            gdf_measurements = self.aggregate_features_by_polygon(
                self.measurements, self.polygons
            )
            self.geo_feats.append(
                np.array(gdf_measurements.drop(columns=['geometry']))
            )

        self.geo_feats = np.concatenate(self.geo_feats, axis=1)
        scaler = StandardScaler()
        self.geo_feats = scaler.fit_transform(self.geo_feats)

        self.geo_feats = GeoDataFrame(
            geometry=self.polygons,
            data={i: list(self.geo_feats[:, i])
                  for i in range(self.geo_feats.shape[1])}
        )

        # add spatial features to instances
        self.instances = self.append_geo_features(self.instances).reset_index(drop=True)
        self.val_instances = self.filter_instances(self.val_instances)
        self.val_instances = self.append_geo_features(self.val_instances).reset_index(drop=True)

        if self.test_instances is not None:
            self.test_instances = self.filter_instances(self.test_instances)
            self.test_instances = self.append_geo_features(self.test_instances)

        # clear and reinitialize data structures
        self.clear_memory()

        # assign instances to polygons
        self.instance_assignments = self.assign_instance_dict(
            copy.deepcopy(self.instances),
            copy.deepcopy(self.polygons)
        )
        self.val_instance_assignments = self.assign_instance_dict(
            copy.deepcopy(self.val_instances),
            copy.deepcopy(self.polygons)
        )

        # touching hexagon pairs
        touching_gdf = gpd.sjoin(
            GeoDataFrame(geometry=self.polygons),
            GeoDataFrame(geometry=self.polygons),
            predicate="touches"
        )
        touching_pairs = list(set(
            tuple(sorted((i, j)))
            for i, j in zip(touching_gdf.index, touching_gdf.index_right)
        ))
        for p_ix in range(len(self.polygons)):
            self.polygon_states.append(1)

        # self-pairs
        for p_ix in range(len(self.polygons)):
            score, length, model = train_polygon_model(self, [p_ix])
            self.models.append(model)
            self.touching_pairs.append((p_ix, p_ix))
            self.score.append(score)
            self.length.append(length)

        # touching pairs
        for pair in touching_pairs:
            self.touching_pairs.append(pair)
            score, length, _ = train_polygon_model(self, list(pair))
            self.score.append(score)
            self.length.append(length)

        # calculate score improvement for each pair
        for pair in self.touching_pairs:
            self.score_diff.append(self.get_score_diff(pair))

        # build adjacency graph
        for key, value in self.touching_pairs:
            self.polygon_neighbors[key].add(value)
            self.polygon_neighbors[value].add(key)
        self.polygon_neighbors = {k: sorted(v) for k, v in self.polygon_neighbors.items()}

    def clear_memory(self):
        self.touching_pairs.clear()
        self.length.clear()
        self.score.clear()
        self.models.clear()
        self.instance_assignments.clear()
        self.val_instance_assignments.clear()
        self.score_diff.clear()
        self.history.clear()
        self.polygon_states.clear()
        self.polygon_neighbors = defaultdict(set)

    def aggregate_features_by_polygon(
        self,
        measurements: List[GeoDataFrame],
        polygons: List[Polygon]
    ) -> GeoDataFrame:
        gdf_bins = GeoDataFrame(
            geometry=polygons,
            data={'polygon_ix': list(range(len(polygons)))}
        )

        all_feature_cols = set()
        for gdf in measurements:
            all_feature_cols.update(gdf.columns.difference(['geometry']))

        # standardize columns across all GeoDataFrames
        standardized_gdfs = []
        for gdf in measurements:
            for col in all_feature_cols:
                if col not in gdf.columns:
                    gdf[col] = None
            standardized_gdfs.append(gdf)

        # combine and spatially join with polygons
        gdf_combined = gpd.GeoDataFrame(
            pd.concat(standardized_gdfs, ignore_index=True),
            crs=measurements[0].crs
        )
        gdf_joined = gpd.sjoin(
            gdf_combined, gdf_bins,
            predicate='intersects', how='left'
        )

        # aggregate by taking mean within each polygon
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
        filter instances to only those within the study area boundary.
        """
        inside_mask = instances['geometry'].apply(lambda x: self.boundary.contains(x))
        num_outside = len(inside_mask) - sum(inside_mask)

        if num_outside > 0:
            warnings.warn(
                f"{num_outside} instances fall outside the boundary and will be removed"
            )

        return instances[inside_mask]

    def append_geo_features(self, instances: GeoDataFrame) -> GeoDataFrame:
       
        instances = gpd.sjoin(
            instances, self.geo_feats,
            predicate='intersects', how='left'
        )
        instances = instances.drop(
            columns=[col for col in instances.columns if 'index' in str(col)]
        )

        # add offset features (position within hexagon)
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
        assign each instance to its containing polygon.
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

    def get_valid_merge_pairs(self) -> Tuple[List, List]:
        
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

    def relative_improvement(self, idx: int) -> float:
        improvement = self.score_diff[idx]
        pooled_separate = improvement + self.score[idx]
        denom = abs(pooled_separate)
        if denom < 1e-12:
            return 0.0
        return improvement / denom

    def merge_polygons(
        self,
        apply_threshold: bool = True
    ) -> bool:

        _, active_indices = self.get_active_polygons()
        self.history.append(active_indices)

        if len(active_indices) <= self.min_polygons:
            print(f"  Reached minimum polygon count ({self.min_polygons}), stopping merges.")
            return False

        # select best pair
        merge_pair, pair_idx = self.get_merge_polygon_pairs()

        if merge_pair[0] == merge_pair[1]:
            return False  # No valid merge

        # reduces the loss, stop rather than keep merging marginal pairs.
        if apply_threshold:
            rel = self.relative_improvement(pair_idx)
            if rel < self.merge_threshold:
                print(f"  Best relative improvement ({rel:.4%}) below threshold "
                      f"({self.merge_threshold:.4%}), stopping.")
                return False

        return self._execute_merge(merge_pair)

    def merge_polygons_forced(self, merge_pair: Tuple[int, int]) -> bool:
        _, active_indices = self.get_active_polygons()
        self.history.append(active_indices)

        if merge_pair[0] == merge_pair[1]:
            return False

        return self._execute_merge(merge_pair)

    def _execute_merge(self, merge_pair: Tuple[int, int]) -> bool:
        
        # create new merged polygon
        new_polygon_index = len(self.polygons)
        self.polygons.append(
            unary_union([self.polygons[merge_pair[0]],
                        self.polygons[merge_pair[1]]])
        )

        # merge instance assignments
        self.instance_assignments.append(
            self.instance_assignments[merge_pair[0]] +
            self.instance_assignments[merge_pair[1]]
        )
        self.val_instance_assignments.append(
            self.val_instance_assignments[merge_pair[0]] +
            self.val_instance_assignments[merge_pair[1]]
        )

        self.polygon_states[merge_pair[0]] = 0
        self.polygon_states[merge_pair[1]] = 0
        self.polygon_states.append(1)  # New polygon is active

        new_neighbors = sorted(list(
            set(self.polygon_neighbors[merge_pair[0]] +
                self.polygon_neighbors[merge_pair[1]])
            - {merge_pair[0], merge_pair[1]}
            | {new_polygon_index}
        ))

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

        for neighbor in new_neighbors[:-1]:  # Exclude self
            self.touching_pairs.append((neighbor, new_polygon_index))
            score, length, model = train_polygon_model(
                self, [neighbor, new_polygon_index]
            )
            self.score.append(score)
            self.length.append(length)
            self.score_diff.append(self.get_score_diff((neighbor, new_polygon_index)))
            self.polygon_neighbors[neighbor].append(new_polygon_index)

        # remove old polygon data
        self.drop_old_polygons(merge_pair[0])
        self.drop_old_polygons(merge_pair[1])

        return True

    def drop_old_polygons(self, old_idx: int):

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

    def simulated_annealing_step(
        self,
        temperature: float
    ) -> bool:
        
        # check minimum polygon constraint first
        _, active_indices = self.get_active_polygons()
        if len(active_indices) <= self.min_polygons:
            print(f"  Reached minimum polygon count ({self.min_polygons}), stopping merges.")
            return False

        _, valid_indices = self.get_valid_merge_pairs()  # exclude self pair
        if not valid_indices:
            return False

        # greedy best pair — used for the SA delta and the exploitation step.
        best_idx = valid_indices[int(np.argmax([self.score_diff[i] for i in valid_indices]))]

        if random.random() < 0.3:
            random_idx = random.choice(valid_indices)
            random_score = self.score_diff[random_idx]
            delta = random_score - self._sa_current_score
            if delta > 0 or random.random() < np.exp(delta / (temperature + 1e-10)):
                self._sa_current_score = random_score
                merge_pair = self.touching_pairs[random_idx]
                return self.merge_polygons_forced(merge_pair)
        self._sa_current_score = self.score_diff[best_idx]
        return self.merge_polygons(apply_threshold=False)


    def construct_clustering(
        self,
        max_iter: int = 10000,
        patience: int = 100,
        restart: bool = True,
        use_simulated_annealing: bool = False,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.98,
        min_temp: float = 0.001
    ) -> Tuple[List[float], List[float], List[float]]:
        
        val_score_list = []
        score_list = []  # track the actual scoring metric used

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # initialize hexagons
        if restart:
            if not self._constructor:
                raise ValueError("No constructor initialized")

            # use cached polygons and features if available
            cache_path = os.path.join(self.save_path, f"spatial_cache_res{self.resolution}_{self.min_samples_per_hexagon}_seed{self.random_seed}.pkl")

            if os.path.exists(cache_path):
                print(f"Loading cached polygons and features (resolution={self.resolution}_{self.min_samples_per_hexagon}, seed={self.random_seed})...")
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

            _, active_indices = self.get_active_polygons()
            sample_counts = [len(self.instance_assignments[i]) for i in active_indices]
            print(f"Initialized with {len(active_indices)} hexagons")
            print(f"  Mean samples per hexagon: {np.mean(sample_counts):.1f}")
            print(f"  Min samples: {np.min(sample_counts)}, Max samples: {np.max(sample_counts)}\n")

        # merging
        merges = 0
        tol = 0  # patience
        temperature = initial_temp
        self._sa_current_score = 0.0  # tracks last executed merge's score_diff for SA delta

        print(f'Merging begins... (scoring method: {self.scoring_method})')

        while True:
           
            try:
                if use_simulated_annealing:
                    
                    merge_success = self.simulated_annealing_step(
                        temperature
                    )
                    temperature *= cooling_rate

                    if temperature < min_temp:
                        print('Temperature reached minimum, stopping.')
                        break
                else:
                   
                    merge_success = self.merge_polygons()

                if not merge_success:
                    print('No available polygons to merge.')
                    break

            except ValueError as e:
                print(f'Error during merge: {e}')
                break

            # evaluate on validation set
            merges += 1
            val_score, _, _, val_uncertainty = self.validate()
            val_score_list.append(val_score)
            current_score = val_score
            score_list.append(current_score)

            # check if this is the best model so far
            is_best = current_score < self.best_score

            # save the best model
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

            # stopping criteria
            if max_iter > 0 and merges >= max_iter:
                print(f'Reached maximum iterations ({max_iter}).')
                break

            if tol >= patience:
                print(f'Patience exhausted ({patience} iterations without improvement).')
                break


        print(f'Clustering complete. Total merges: {merges}')
        print(f'Best {self.scoring_method}: {self.best_score:.4f}')

        return val_score_list, score_list

    def get_instances_in_polygon(
        self,
        polygon_idx: int,
        dict_instance: Optional[List] = None,
        instances_used: Optional[GeoDataFrame] = None
    ) -> GeoDataFrame:
        
        if dict_instance is None:
            dict_instance = self.instance_assignments
            instances_used = self.instances

        if polygon_idx >= len(self.polygons) or self.polygons[polygon_idx] is None:
            raise ValueError(f"No polygon with index {polygon_idx}")

        return instances_used.iloc[list(dict_instance[polygon_idx])]

    def get_halo_instances(self, polygon_idxs: List[int]) -> GeoDataFrame:

        region = unary_union([self.polygons[i] for i in polygon_idxs])
        ring = region.buffer(self.halo_buffer).difference(region)
        if ring.is_empty:
            return self.instances.iloc[[]]

        pos = self.instances.sindex.query(ring, predicate='intersects')
        halo = self.instances.iloc[pos]

        own = set().union(*(set(self.instance_assignments[i]) for i in polygon_idxs))
        return halo[~halo.index.isin(own)]

    def get_within_polygons_index(self, instance: GeoDataFrame) -> int:
        
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
        _, active_ix = self.get_active_polygons()

        models = []
        X_list = []
        y_list = []

        # collect data from each active polygon
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

        # get predictions and uncertainties
        preds_list = []
        uncertainty_list = []

        for X, model in zip(X_list, models):
            pred, pred_std = model.predict(X, return_std=True)
            preds_list.append(pred)
            uncertainty_list.extend(pred_std)
       
        y_mean = np.array([item for sublist in preds_list for item in sublist])
        y = np.array(y_list)
        y_std = np.array(uncertainty_list)

        if self.scoring_method == 'combined_score':
            score = combined_score(y, y_mean, y_std, self.uncertainty_weight)
        elif self.scoring_method == 'beta_nll':
            score = gaussian_beta_nll(y, y_mean, y_std, beta=self.beta)
        elif self.scoring_method == 'mse':
            score = mean_squared_error(y, y_mean)
        else:
            raise KeyError("scoring method is not applicable!")

        return score, y_mean, self.val_instance_assignments, y_std

    def predict(self, instances: GeoDataFrame) -> Tuple[float, List[float], GeoDataFrame, List[float]]:
       
        # Filter and add spatial features
        instances = self.filter_instances(instances)
        instances = self.append_geo_features(instances)

        preds = []
        uncertainties = []
        for i in range(len(instances)):
            # filter containing polygon
            polygon_idx = self.get_within_polygons_index(instances.iloc[i])

            # extract features
            X = instances.drop(columns=["geometry", "label"]).iloc[i:i+1].values

            model = self.models[polygon_idx]
            pred, pred_std = model.predict(X, return_std=True)
            preds.append(pred.item())
            uncertainties.append(pred_std.item())

        y = instances['label'].values
        preds_arr = np.array(preds)
        uncertainties_arr = np.array(uncertainties)

        if self.scoring_method == 'combined_score':
            score = combined_score(y, preds_arr, uncertainties_arr, self.uncertainty_weight)
        elif self.scoring_method == 'beta_nll':
            score = gaussian_beta_nll(y, preds_arr, uncertainties_arr, beta=self.beta)
        elif self.scoring_method == 'mse':
            score = mean_squared_error(y, preds_arr)
        else:
            raise KeyError("scoring method is not applicable!")

        return score, preds, instances, uncertainties

    def get_active_polygons(self) -> Tuple[List[Polygon], List[int]]:
       
        active_indices = [
            i for i in range(len(self.polygon_states))
            if self.polygon_states[i] == 1
        ]
        active_polygons = [self.polygons[i] for i in active_indices]

        return active_polygons, active_indices

    def save_best_instance(self,
        best_score: Optional[float] = None,
        best_uncertainty: Optional[float] = None,
        score_name: str = "Score"):
        
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
