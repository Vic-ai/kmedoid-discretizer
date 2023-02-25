from typing import Dict, Optional

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from enum import Enum

import joblib
import numpy as np
import pandas as pd
import ray
from ray.util.joblib import register_ray
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


class ValidatedEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        choices = [e.value for e in cls]
        raise ValueError(
            "%r is not a valid %s, please choose from %s"
            % (value, cls.__name__, choices)
        )


class Strategy(ValidatedEnum):
    RANDOM = "random"
    HEURISTIC = "heuristic"
    K_MEDOIDS_PLUS = "k-medoids++"
    BUILD = "build"


class Encode(ValidatedEnum):
    ORDINAL = "ordinal"
    ONEHOT_DENSE = "onehot-dense"


class Backend(ValidatedEnum):
    SERIAL = "serial"
    MULTIPROCESSING = "multiprocessing"
    RAY = "ray"


@dataclass
class RayConfig:
    address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    resources: Optional[Dict] = None


@contextmanager
def backend_context_manager(backend: Backend, ray_config: RayConfig, n_jobs: int):
    if backend == Backend.RAY:
        logger.info(f"Init {backend}")
        ray.init(**asdict(ray_config))
        register_ray()
        n_jobs = min(int(ray.available_resources()["CPU"]), n_jobs)
    elif backend == Backend.SERIAL:
        n_jobs = 1

    with joblib.parallel_backend("ray") if backend == Backend.RAY else nullcontext():
        try:
            yield n_jobs
        finally:
            if backend == Backend.RAY:
                ray.shutdown()

    if backend == Backend.RAY:
        logger.info(f"Shutdown {backend}")
        ray.shutdown()


def auto_compute_n_bins_density_peaks(data: pd.Series) -> int:
    kde = gaussian_kde(data)
    no_samples = 10000
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    peaks, _ = find_peaks(probs, height=0)
    return len(peaks) + 1
