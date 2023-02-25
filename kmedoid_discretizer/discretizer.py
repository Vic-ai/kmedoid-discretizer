from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging
import traceback
import warnings

import numpy as np
import pandas as pd
import ray
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn_extra.cluster import KMedoids

from kmedoid_discretizer.utils.utils_discretizer import (
    Backend,
    Encode,
    RayConfig,
    Strategy,
    auto_compute_n_bins_density_peaks,
    backend_context_manager,
)

warnings.filterwarnings("ignore", module="sklearn")
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class KmedoidDiscretizer(TransformerMixin, BaseEstimator):
    """
    KmedoidDiscretizer (Adaptative Kmedoid discretizer) allows to discritize numerical feature into `n_bins` using Kmedoids Clustering algrorithm from sklearn.
    With this implemenation, we can have:
    - A custom number of bins for each numeral feature. Kmedoids will be run for each columns.
    - Adapt the number of bins dynamically whenever this one is two high (more precesly when two centroids are assigned to the same data point.)
    - Multiple Backends are possible: serial, multiprocessing, and ray to speed up the Kmedoids compuation.
    - Mainly use Pandas DataFrame and Numpy array.
    """

    def __init__(
        self,
        n_bins: Union[List[int], str, int] = "auto",
        encode: Literal[Encode.ONEHOT_DENSE, Encode.ORDINAL] = Encode.ORDINAL,
        strategy: Literal[
            Strategy.K_MEDOIDS_PLUS, Strategy.BUILD, Strategy.HEURISTIC, Strategy.RANDOM
        ] = Strategy.K_MEDOIDS_PLUS,
        backend: Literal[
            Backend.MULTIPROCESSING, Backend.RAY, Backend.SERIAL
        ] = Backend.MULTIPROCESSING,
        ray_address: Optional[str] = None,
        ray_num_cpus: Optional[int] = None,
        ray_num_gpus: Optional[int] = None,
        ray_resources: Optional[Dict] = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        """
        KmedoidDiscretizer __init__ method.

        Args:
            n_bins (Union[List[int], str, int]): number of bins for each/all feature columns. Default is "auto" (compute automatically n_bins based on input data).
            encode (str): Encoding of the output data. Default ordinal.
            strategy (str): Initial strategy for Kmedoid training. Default is k-medoids++.
            backend (str): Backend used to train Kmedoid model(s). Default is multiprocessing.
            ray_address (Optional[str]): address in ray init(), referring to IP address to connect to a ray cluster. See ray documentation. Default is None.
            ray_num_cpus (Optional[int]): num_cpus in ray init(), referring to the number of CPU in ray cluster. See ray documentation. Default is None.
            ray_num_gpus (Optional[int]): num_gpus in ray init(), referring to the number of GPU in ray cluster. See ray documentation. Default is None.
            ray_resources (Optional[Dict]): resources in ray init(), referring to ray cluster resources configuration. See ray documentation. Default is None.
            seed (int): Random seed initialization for reproducibility. Default 0.
            verbose (bool): Allow more detailed logging. Default False.
        """
        self.strategy = Strategy(strategy)
        self.encode = Encode(encode)
        self.backend = Backend(backend)

        self.ray_address = ray_address
        self.ray_num_cpus = ray_num_cpus
        self.ray_num_gpus = ray_num_gpus
        self.ray_resources = ray_resources

        self.ray_config = RayConfig(
            address=ray_address,
            num_cpus=ray_num_cpus,
            num_gpus=ray_num_gpus,
            resources=ray_resources,
        )

        if isinstance(n_bins, str):
            if n_bins != "auto":
                raise ValueError(
                    f"Invalid n_bins {n_bins}. Use 'auto', an integer or a list of interger."
                )

        self.n_bins: Union[List[int], str, int] = n_bins
        self.ohe: Optional[OneHotEncoder] = None
        self.kmedoids: Dict[str, KMedoids] = {}
        self.was_ray_initialized = False
        self.verbose = verbose
        self.seed = seed

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> KmedoidDiscretizer:
        """Fit the estimator.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input dataset.
            y (Optional[Union[pd.DataFrame, np.ndarray]]): Label dataset. Default None.

        Returns:
            KmedoidDiscretizer: KmedoidDiscretizer object containing kmedoids object for all features.
        """
        # Validate Data
        df = self._validate_X(X)
        self._validate_n_bins(df)

        df = df.copy()
        self.columns = df.columns.to_list()

        # Run Kmedoid for each feature
        n_jobs = len(self.columns)

        cols, kmedoids = [], []
        try:
            with backend_context_manager(
                backend=self.backend, ray_config=self.ray_config, n_jobs=n_jobs
            ) as n_jobs:
                if self.verbose:
                    self.was_ray_initialized = ray.is_initialized()
                    logger.info(f"Is ray initialized: {self.was_ray_initialized}.")

                cols, kmedoids = zip(
                    *Parallel(n_jobs=n_jobs)(
                        delayed(self._fit_feature)(
                            col, df[col], self.n_bins[i], self.strategy
                        )
                        for i, col in enumerate(self.columns)
                    )
                )
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Fitting error - {e}")

        self.kmedoids = {col: kmedoid for col, kmedoid in zip(cols, kmedoids)}

        if not self.kmedoids:
            raise ValueError("Fitting failed.")

        # Encoding
        if self.encode == Encode.ONEHOT_DENSE:
            df_transform = self.transform(df)
            self.ohe = OneHotEncoder()
            self.ohe.fit(df_transform)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Discretize the data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input dataset.

        Returns:
            pd.DataFrame: Transformed (discritized) data.
        """
        # Make it a dataframe
        df = self._validate_X(X)

        if len(df.columns) != len(self.columns):
            raise ValueError(
                f"X_test must be have the same number of feature as X_train. {len(df.columns)} != {len(self.columns)}"
            )

        df = df.copy()
        for col in self.columns:
            if col in self.kmedoids:
                df[col] = self.kmedoids[col].predict(
                    np.expand_dims(df[col].to_numpy(), -1)
                )
        if self.ohe:
            return pd.DataFrame(self.ohe.transform(df).toarray()).reset_index()
        return df

    def _validate_n_bins(self, df: pd.DataFrame) -> None:
        """
        Validate (adjust) n_bins_, the number of bins per feature.

        Args:
            df (pd.DataFrame): Input dataset.
        """
        assert df.empty == False, "Cannot provide an empty dataframe df."

        if isinstance(self.n_bins, str):
            try:
                self.n_bins = []
                for col in df.columns:
                    self.n_bins.append(auto_compute_n_bins_density_peaks(df[col]))
            except Exception as e:
                raise ValueError(f"auto_compute_n_bins_density_peaks error - {e}")

        if isinstance(self.n_bins, int):
            self.n_bins = [self.n_bins] * df.shape[1]

        if len(self.n_bins) != df.shape[1]:
            raise ValueError(
                f"n_bins must be a list of size of the number of feature in df. {len(self.n_bins)} != {df.shape[1]}."
            )

    def _validate_X(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """_validate_X transform and validate inpt data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input dataset.

        Raises:
            TypeError: X must be a pd.DataFrame or np.ndarray. Type {type(X)} is not supported.

        Returns:
            pd.DataFrame: Input dataset as Pandas Dataframe.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f"_feature_{col}" for col in range(X.shape[1])]
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pd.DataFrame or np.ndarray. Type {type(X)} is not supported."
            )
        return X

    def _fit_feature(
        self, col: str, df_col: pd.Series, n_bins: int, strategy: Strategy
    ) -> Tuple[str, KMedoids]:
        """_fit_feature compute kmedoids over a column (pd.Series), remove duplicate clusters and return the Kmedoids object alongside the column name.

        Args:
            col (str): Pandas Series column name, also known as the feature name.
            df_col (pd.Series): Pandas Series input data to be discritized.
            n_bins (int): Number of bins.
            strategy (Strategy): Initial strategy for Kmedoid training.

        Returns:
            Union[str, KMedoids]: feature/column name and trained/optimized KMedoids object.
        """
        kmedoid = KMedoids(n_bins, init=strategy.value, random_state=self.seed)
        kmedoid.fit(np.expand_dims(df_col.to_numpy(), -1))
        # Remove duplicate bins to use optimal number of bins
        centroids_index_to_keep = np.sort(
            np.unique(kmedoid.cluster_centers_, return_index=True)[1]
        )
        kmedoid.cluster_centers_ = np.expand_dims(
            np.take(kmedoid.cluster_centers_, centroids_index_to_keep), -1
        )
        if self.verbose and n_bins != len(centroids_index_to_keep):
            logger.info(
                f"Feature {col} is not well clustered with {n_bins} bins, it has been replaced by {len(centroids_index_to_keep)} bins."
            )
        return col, kmedoid
