from typing import List, Optional
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMOnlineLoader
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from shapely.ops import unary_union
import geopandas as gpd
import random
import torch

class Constructor:
    """Base class for polygon creation strategies."""

    def construct(
        self,
        instances: Optional[GeoDataFrame] = None,
        remove_empty: bool = False,
        min_samples: int = 20
    ) -> List[int]:
        polygons, feats = self._create_polygons(instances)
        print(f"Created {len(polygons)} polygons")
        polygons_retained = []
        feats_retained = []
        instance_points = unary_union(instances.geometry)
        if remove_empty and instances is not None:
            filtered = [(poly, feat) for poly, feat in zip(polygons, feats) if poly.intersects(instance_points)]
        else:
            filtered = [(poly, feat) for poly, feat in zip(polygons, feats) if True]
        polygons_retained, feats_retained = zip(*filtered) if filtered else ([], [])

        # Merge small/empty hexagons before filtering
        if instances is not None and min_samples > 0:
            polygons_merged, feats_merged = self._merge_insufficient_hexagons(
                list(polygons_retained), np.array(feats_retained), instances, min_samples
            )
        print(f"Retained {len(polygons_merged)} polygons")
        print(f"Retained {len(feats_merged)} feats")
        return list(polygons_merged), np.array(feats_merged)

    def _merge_insufficient_hexagons(
        self,
        polygons: List[Polygon],
        feats: np.ndarray,
        instances: GeoDataFrame,
        min_samples: int
    ) -> tuple[List[Polygon], np.ndarray]:
        """
        Merge hexagons with fewer than min_samples instances.
        Uses geographic proximity (closest centroid) for merging.
        """
        # Assign instances to polygons
        gdf_polygons = GeoDataFrame(geometry=polygons)
        assignments = self._assign_instances_to_polygons(instances, gdf_polygons)

        # Track active polygons (1=active, 0=merged)
        polygon_states = [1] * len(polygons)

        # Build neighbor graph
        neighbors = self._build_neighbor_graph(gdf_polygons)

        num_merges = 0
        initial_small_count = sum(1 for a in assignments if len(a) < min_samples)

        if initial_small_count > 0:
            print(f"Merging {initial_small_count} hexagons with < {min_samples} samples...")

        while True:
            # Find hexagons with insufficient samples
            small_hexagons = [
                idx for idx in range(len(polygons))
                if polygon_states[idx] == 1 and len(assignments[idx]) < min_samples
            ]

            if not small_hexagons:
                break

            # Select smallest hexagon (deterministic)
            target_idx = min(small_hexagons, key=lambda idx: (len(assignments[idx]), polygons[idx].centroid.x))

            # Find active neighbors
            active_neighbors = [
                n for n in neighbors[target_idx]
                if polygon_states[n] == 1
            ]

            if not active_neighbors:
                # Isolated hexagon - mark inactive
                polygon_states[target_idx] = 0
                print(f"  Warning: Hexagon {target_idx} has no neighbors - marking inactive")
                continue

            # Find closest neighbor by centroid distance
            neighbor_idx = min(
                active_neighbors,
                key=lambda n: polygons[target_idx].centroid.distance(polygons[n].centroid)
            )

            # Merge geometries and assignments
            new_polygon = unary_union([polygons[target_idx], polygons[neighbor_idx]])
            new_feat = (feats[target_idx] + feats[neighbor_idx]) / 2  # Average features
            new_assignment = assignments[target_idx] + assignments[neighbor_idx]

            # Add merged polygon
            polygons.append(new_polygon)
            feats = np.vstack([feats, new_feat])
            assignments.append(new_assignment)
            polygon_states.append(1)

            # Update states
            polygon_states[target_idx] = 0
            polygon_states[neighbor_idx] = 0

            # Update neighbors
            new_idx = len(polygons) - 1
            new_neighbors = (neighbors[target_idx] | neighbors[neighbor_idx]) - {target_idx, neighbor_idx}

            for n in new_neighbors:
                neighbors[n].discard(target_idx)
                neighbors[n].discard(neighbor_idx)
                neighbors[n].add(new_idx)

            neighbors[new_idx] = new_neighbors

            num_merges += 1

            if num_merges % 50 == 0:
                print(f"  Merged {num_merges} hexagons...")

        if num_merges > 0:
            print(f"  Total merges: {num_merges}\n")

        # Return only active polygons
        active_polygons = [p for i, p in enumerate(polygons) if polygon_states[i] == 1]
        active_feats = feats[[i for i in range(len(polygons)) if polygon_states[i] == 1]]

        return active_polygons, active_feats

    def _assign_instances_to_polygons(
        self,
        instances: GeoDataFrame,
        polygons: GeoDataFrame
    ) -> List[List[int]]:
        """Assign instance indices to polygons."""
        joined = gpd.sjoin(polygons, instances, how="left", predicate="contains")
        assignments = (
            joined.groupby(joined.index)
            .apply(lambda x: x.index_right.dropna().astype(int).tolist())
            .reindex(range(len(polygons)), fill_value=[])
            .tolist()
        )
        return assignments

    def _build_neighbor_graph(self, polygons: GeoDataFrame) -> dict:
        """Build adjacency graph for polygons."""
        touching_gdf = gpd.sjoin(polygons, polygons, predicate="touches")
        neighbors = {i: set() for i in range(len(polygons))}

        for i, j in zip(touching_gdf.index, touching_gdf.index_right):
            if i != j:
                neighbors[i].add(j)

        return neighbors

    def _create_polygons(self, instances: Optional[GeoDataFrame]) -> List[Polygon]:
        """Create polygons based on the points."""
        raise NotImplementedError(
            "Constructor subclasses must implement _create_polygons()"
        )

class SraiConstructor(Constructor):
    """Creates polygons based on a SRAI regionalizer."""
    def __init__(self,
                 selected_area: str,
                 resolution: int = 9,
                 encoder_sizes: Optional[list[int]] = [10, 5],
                 add_offset_features: bool = True,
                 min_samples_per_hexagon: int = 20,
                 random_seed: int = 42
                 ):
        self.area = geocode_to_region_gdf(selected_area)
        self.loader = OSMOnlineLoader()
        self.joiner = IntersectionJoiner()
        self.regionalizer = H3Regionalizer(resolution=resolution)
        self.embedder = Hex2VecEmbedder(encoder_sizes)
        self.add_offset_features = add_offset_features
        self.min_samples_per_hexagon = min_samples_per_hexagon
        self.random_seed = random_seed

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        import pytorch_lightning as pl
        # workers=True ensures dataloader workers are also seeded
        pl.seed_everything(self.random_seed, workers=True)

        # Additional manual seeding for completeness
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        # Enable deterministic algorithms for CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force deterministic algorithms (may raise error on non-deterministic ops)
        # Set warn_only=True to get warnings instead of errors
        torch.use_deterministic_algorithms(True, warn_only=True)
        
    def _create_polygons(self, instances: Optional[GeoDataFrame], query={
                                                        "leisure": "park",
                                                        "railway": "station",
                                                        "amenity": "school",
                                                        "amenity": "university",
                                                        "building": "supermarket",
                                                    }) -> List[Polygon]:
        # Set seed for reproducibility
        # self._set_seed()

        # area
        area_limited = MultiPoint(list(instances.geometry)).convex_hull
        features_gdf = self.loader.load(area_limited, query)
        self.area.geometry = [area_limited]
        regions = self.regionalizer.transform(self.area)
        joint = self.joiner.transform(regions, features_gdf)

        # generate embeddings with error handling
        try:
            self._set_seed()  # Set seed again before neural network training
            neighbourhood = H3Neighbourhood(regions_gdf=regions)

            # Configure trainer for deterministic behavior
            trainer_kwargs = {
                "deterministic": True,  # Force deterministic algorithms in PyTorch Lightning
                "enable_progress_bar": False,
            }

            # Try to fit the embedder with deterministic trainer settings
            self.embedder.fit(
                regions, features_gdf, joint, neighbourhood,
                batch_size=128,
                trainer_kwargs=trainer_kwargs
            )
            embeddings = self.embedder.transform(regions, features_gdf, joint)
        except (TypeError, AttributeError, ImportError) as e:
            print(f"Warning: Hex2Vec embedding failed ({e}). Using fallback feature extraction.")
            # Fallback: Use simple feature counts as embeddings
            embeddings = self._create_fallback_embeddings(regions, features_gdf, joint)

        # Store regions for offset calculation
        self.regions = regions

        return regions.geometry, np.array(embeddings)

    def _create_fallback_embeddings(self, regions, features_gdf, joint):
        """
        Create simple embeddings based on feature counts when Hex2Vec fails.
        This is a fallback for compatibility issues with SRAI/PyTorch Lightning.
        """
        import warnings
        warnings.warn("Using fallback embeddings. Consider updating srai and pytorch-lightning packages.")

        # Count features per region
        n_regions = len(regions)

        # Get feature counts from joint (regions-features intersection)
        if hasattr(joint, 'groupby'):
            # Joint is a DataFrame with region indices
            feature_counts = joint.groupby(joint.index).size()
            embeddings = np.zeros((n_regions, 10))  # Match expected embedding size

            for idx in range(n_regions):
                if idx in feature_counts.index:
                    # Use feature count as first dimension
                    embeddings[idx, 0] = feature_counts[idx]

            # Normalize
            embeddings = embeddings / (embeddings.max() + 1e-10)
        else:
            # If no joint data, use zero embeddings
            embeddings = np.zeros((n_regions, 10))

        return embeddings

    def compute_offset_features(self, data: GeoDataFrame, hex_polygons: GeoDataFrame) -> GeoDataFrame:
        """
        Compute position offset features for each node relative to its hexagon centroid to add variation to nodes within the same hexagon.
        """
        if not self.add_offset_features:
            return data

        # Spatial join to assign each node to its hexagon
        data_with_hex = gpd.sjoin(data, hex_polygons, how="left", predicate="within")

        # Compute offset features for each node
        offset_x = []
        offset_y = []
        offset_distance = []
        offset_angle = []

        for idx, row in data_with_hex.iterrows():
            hex_idx = row.get('index_right', None)

            if hex_idx is not None and hex_idx in hex_polygons.index:
                hex_centroid = hex_polygons.loc[hex_idx].geometry.centroid

                # Calculate offset from hex centroid
                dx = row.geometry.x - hex_centroid.x
                dy = row.geometry.y - hex_centroid.y
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)

                offset_x.append(dx)
                offset_y.append(dy)
                offset_distance.append(distance)
                offset_angle.append(angle)
            else:
                # For nodes outside hexagons, use zero offset
                offset_x.append(0.0)
                offset_y.append(0.0)
                offset_distance.append(0.0)
                offset_angle.append(0.0)

        # Add offset features to original data (not the joined version)
        data_copy = data.copy()
        data_copy['offset_x'] = offset_x
        data_copy['offset_y'] = offset_y
        data_copy['offset_distance'] = offset_distance
        data_copy['offset_angle'] = offset_angle

        return data_copy
            
