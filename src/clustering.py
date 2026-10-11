"""
K-Means Clustering & Preprocessing Engine for Customer Segmentation Project.

Features:
1. Log1p Transformation for skewed Frequency and Monetary distributions.
2. StandardScaler normalization so feature magnitudes are directly comparable.
3. Encapsulated RFMClusterPipeline preserving fitted scaler and KMeans model.
4. Cluster optimization evaluation (Elbow / Inertia and Silhouette Score across k in [2, 8]).
5. Dual Engine Architecture: Native Scikit-Learn with Pure NumPy Fallback to run seamlessly
   even on systems where Windows Application Control Policy blocks C-extension DLLs (_sag_fast.pyd).
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd


# Pure NumPy implementations for environments with restricted DLL loading
class PureNumPyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X_arr - self.mean_) / self.scale_

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted yet.")
        return (X_arr - self.mean_) / self.scale_


class PureNumPyKMeans:
    def __init__(self, n_clusters: int = 4, random_state: int = 42, max_iter: int = 300, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = 0.0

    def fit_predict(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"Dataset contains {n_samples} sample(s), which is fewer than requested {self.n_clusters} clusters.")

        best_inertia = float("inf")
        best_labels = None
        best_centers = None

        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx if self.random_state is not None else None
            rng = np.random.RandomState(seed)

            # Random initialization
            init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X_arr[init_indices].copy()

            labels = np.zeros(n_samples, dtype=int)
            for _ in range(self.max_iter):
                # Calculate distances
                dists = np.linalg.norm(X_arr[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                new_labels = np.argmin(dists, axis=1)

                if np.array_equal(labels, new_labels):
                    break
                labels = new_labels

                # Update centers
                for c in range(self.n_clusters):
                    members = X_arr[labels == c]
                    if len(members) > 0:
                        centers[c] = np.mean(members, axis=0)
                    else:
                        centers[c] = X_arr[rng.choice(n_samples)]

            dists = np.linalg.norm(X_arr[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            inertia = float(np.sum(min_dists ** 2))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        return self.labels_


def pure_numpy_silhouette_score(X: Any, labels: Any) -> float:
    X_arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) <= 1 or len(unique_labels) >= len(X_arr):
        return 0.0

    if len(X_arr) > 400:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_arr), 400, replace=False)
        X_arr = X_arr[idx]
        labels = labels[idx]
        unique_labels = np.unique(labels)

    n_samples = len(X_arr)
    sil_scores = np.zeros(n_samples)

    diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    for i in range(n_samples):
        c_i = labels[i]
        same_mask = (labels == c_i)
        same_mask[i] = False

        if np.sum(same_mask) == 0:
            sil_scores[i] = 0.0
            continue

        a_i = np.mean(dist_matrix[i, same_mask])

        b_i = float("inf")
        for c_other in unique_labels:
            if c_other == c_i:
                continue
            other_mask = (labels == c_other)
            if np.sum(other_mask) > 0:
                dist_other = np.mean(dist_matrix[i, other_mask])
                if dist_other < b_i:
                    b_i = dist_other

        if b_i == float("inf"):
            sil_scores[i] = 0.0
        else:
            max_ab = max(a_i, b_i)
            sil_scores[i] = (b_i - a_i) / max_ab if max_ab > 0 else 0.0

    return float(np.mean(sil_scores))


# Safe Scikit-Learn Import with Pure NumPy Fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    # Test loading native DLL to ensure Application Control policy won't block execution at runtime
    _test_scaler = StandardScaler()
    _test_km = KMeans(n_clusters=2, n_init=1)
    _test_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    _test_km.fit(_test_matrix)
except Exception:
    # Application Control Policy or DLL load block triggered -> Seamless Pure NumPy fallback
    StandardScaler = PureNumPyStandardScaler
    KMeans = PureNumPyKMeans
    silhouette_score = pure_numpy_silhouette_score


MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
DEFAULT_CLUSTERS = 4
DEFAULT_RANDOM_STATE = 42


class RFMClusterPipeline:
    """
    Encapsulates preprocessing (log transform + standardization) and K-Means model.
    Keeps fitted scaler and cluster model together to guarantee consistent transformation.
    """
    
    def __init__(self, n_clusters: int = DEFAULT_CLUSTERS, random_state: int = DEFAULT_RANDOM_STATE):
        if not isinstance(n_clusters, int) or n_clusters < MIN_CLUSTERS or n_clusters > MAX_CLUSTERS:
            raise ValueError(f"Number of clusters (n_clusters) must be an integer between {MIN_CLUSTERS} and {MAX_CLUSTERS}. Got: {n_clusters}")
            
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.is_fitted: bool = False

    def transform_features(self, rfm_df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        required = ["Recency", "Frequency", "Monetary"]
        missing = [col for col in required if col not in rfm_df.columns]
        if missing:
            raise ValueError(f"RFM DataFrame is missing required columns: {missing}")

        features = pd.DataFrame(index=rfm_df.index)
        features["Recency"] = rfm_df["Recency"].astype(float)
        features["Frequency"] = np.log1p(np.maximum(rfm_df["Frequency"].astype(float), 0))
        features["Monetary"] = np.log1p(np.maximum(rfm_df["Monetary"].astype(float), 0))

        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_matrix = self.scaler.fit_transform(features)
        else:
            scaled_matrix = self.scaler.transform(features)

        return scaled_matrix

    def fit_predict(self, rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        if rfm_df is None or rfm_df.empty:
            raise ValueError("Input RFM DataFrame is empty or None.")

        n_samples = len(rfm_df)
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Dataset contains {n_samples} customer(s), which is fewer than the requested {self.n_clusters} clusters. "
                f"Please reduce cluster count or provide more data."
            )

        scaled_matrix = self.transform_features(rfm_df, fit_scaler=True)

        self.model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        cluster_labels = self.model.fit_predict(scaled_matrix)
        self.is_fitted = True

        result_df = rfm_df.copy()
        result_df["Cluster"] = cluster_labels

        return result_df, scaled_matrix


def evaluate_cluster_range(
    rfm_df: pd.DataFrame, 
    k_min: int = MIN_CLUSTERS, 
    k_max: int = MAX_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, List[Any]]:
    results: Dict[str, List[Any]] = {"k": [], "inertia": [], "silhouette": []}
    
    if rfm_df is None or len(rfm_df) < k_min:
        return results

    max_k_possible = min(k_max, len(rfm_df) - 1)
    if max_k_possible < k_min:
        return results

    temp_pipeline = RFMClusterPipeline(n_clusters=k_min, random_state=random_state)
    scaled_matrix = temp_pipeline.transform_features(rfm_df, fit_scaler=True)

    for k in range(k_min, max_k_possible + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        
        inertia_val = float(km.inertia_)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            try:
                sil_val = float(silhouette_score(scaled_matrix, labels))
            except Exception:
                sil_val = 0.0
        else:
            sil_val = 0.0

        results["k"].append(k)
        results["inertia"].append(round(inertia_val, 2))
        results["silhouette"].append(round(sil_val, 4))

    return results
