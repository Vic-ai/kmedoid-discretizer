import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn_extra.cluster import KMedoids

from kmedoid_discretizer.discretizer import KmedoidDiscritizer
from kmedoid_discretizer.utils.utils_discretizer import (
    Backend,
    Encode,
    RayConfig,
    Strategy,
)


class TestKmedoidDiscritizer(unittest.TestCase):
    """
    TestKmedoidDiscritizer test the main components of KmedoidDiscritizer, including:
    - I/O
    - Backend
    - Data Shape
    - main functions
    - Sklearn Pipelines
    """

    def test_kmedoid_discretizer_wrong_backend(self):
        """Test non-existing backend"""
        with self.assertRaises(Exception) as e:
            KmedoidDiscritizer(2, backend="DO_NOT_EXIST")

        self.assertEqual(
            str(e.exception),
            "'DO_NOT_EXIST' is not a valid Backend, please choose from ['serial', 'multiprocessing', 'ray']",
        )

    def test_kmedoid_discretizer_wrong_str_bins(self):
        """Test wrong n_bins input"""
        with self.assertRaises(Exception) as e:
            KmedoidDiscritizer("test")

        self.assertEqual(
            str(e.exception),
            "Invalid n_bins test. Use 'auto', an integer or a list of interger.",
        )

    def test_kmedoid_discretizer_wrong_strategy(self):
        """Test wrong strategy input"""
        with self.assertRaises(Exception) as e:
            KmedoidDiscritizer(strategy="DO_NOT_EXIST")

        self.assertEqual(
            str(e.exception),
            "'DO_NOT_EXIST' is not a valid Strategy, please choose from ['random', 'heuristic', 'k-medoids++', 'build']",
        )

    def test_kmedoid_discretizer_wrong_encode(self):
        """Test wrong strategy encoding"""
        with self.assertRaises(Exception) as e:
            KmedoidDiscritizer(encode="DO_NOT_EXIST")

        self.assertEqual(
            str(e.exception),
            "'DO_NOT_EXIST' is not a valid Encode, please choose from ['ordinal', 'onehot-dense']",
        )

    def test_kmedoid_discretizer_fit_empty_df(self):
        """Test empty input dataframe"""
        x = pd.DataFrame.from_dict({})
        with self.assertRaises(Exception) as e:
            kd = KmedoidDiscritizer()
            kd.fit_transform(x)

        self.assertEqual(str(e.exception), "Cannot provide an empty dataframe df.")

    def test_x_test_and_x_train_same_features(self):
        """Test test data has same feature column numbers as train data"""
        x = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        x_test = pd.DataFrame.from_dict({f"feature_{i}": [1, 4] for i in range(4)})
        discretizer = KmedoidDiscritizer()
        discretizer.fit_transform(x)
        with self.assertRaises(Exception) as e:
            discretizer.transform(x_test)

        self.assertEqual(
            str(e.exception),
            "X_test must be have the same number of feature as X_train. 4 != 3",
        )

    def test_validate_X_wrong_type_list(self):
        """Test input type (list is not supported)"""
        x = [[1, 2], [2, 3]]
        discretizer = KmedoidDiscritizer()
        with self.assertRaises(Exception) as e:
            discretizer._validate_X(x)

        self.assertEqual(
            str(e.exception),
            "X must be a pd.DataFrame or np.ndarray. Type <class 'list'> is not supported.",
        )

    def test_validate_n_bins_unmatch(self):
        """Test invalid number of bins"""
        x = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        discretizer = KmedoidDiscritizer([1, 2])
        with self.assertRaises(Exception) as e:
            discretizer._validate_n_bins(x)

        self.assertEqual(
            str(e.exception),
            "n_bins must be a list of size of the number of feature in df. 2 != 3.",
        )

    def test_onehot_encoding_shape(self):
        """Test onehot encoding output shape"""
        x = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        discretizer = KmedoidDiscritizer(2, encode=Encode.ONEHOT_DENSE)
        X_discrite = discretizer.fit_transform(x)

        assert X_discrite.shape == (3, 7)

    def test_ray_backend(self):
        """Test ray backend"""
        x = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        discretizer = KmedoidDiscritizer(backend="ray", verbose=True)
        discretizer = discretizer.fit(x)

        assert discretizer.was_ray_initialized == True

    def test_fit_failed(self):
        """Test sklearn failure of fit function"""
        X = pd.DataFrame.from_dict(
            {f"feature_{i}": [{"Wrong"}, 2, 3] for i in range(3)}
        )
        discretizer = KmedoidDiscritizer(1)
        with self.assertRaises(Exception) as e:
            discretizer = discretizer.fit(X)
        self.assertEqual(
            str(e.exception),
            "Fitting failed.",
        )

    def test_pre_fit_failed(self):
        """Test auto_compute_n_bins_density_peaks failure of fit function"""
        X = pd.DataFrame.from_dict(
            {f"feature_{i}": [{"Wrong"}, 2, 3] for i in range(3)}
        )
        discretizer = KmedoidDiscritizer()
        with self.assertRaises(Exception) as e:
            discretizer = discretizer.fit(X)

        assert "auto_compute_n_bins_density_peaks error" in str(e.exception)

    def test_fit(self):
        """Test fit function"""
        X = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        discretizer = KmedoidDiscritizer()
        discretizer = discretizer.fit(X)
        expected_discretizer__dict__ = {
            "strategy": Strategy.K_MEDOIDS_PLUS,
            "encode": Encode.ORDINAL,
            "backend": Backend.MULTIPROCESSING,
            "ray_address": None,
            "ray_num_cpus": None,
            "ray_num_gpus": None,
            "ray_resources": None,
            "ray_config": RayConfig(
                address=None, num_cpus=None, num_gpus=None, resources=None
            ),
            "n_bins": [2, 2, 2],
            "ohe": None,
            "kmedoids": {
                "feature_0": KMedoids(init="k-medoids++", n_clusters=2, random_state=0),
                "feature_1": KMedoids(init="k-medoids++", n_clusters=2, random_state=0),
                "feature_2": KMedoids(init="k-medoids++", n_clusters=2, random_state=0),
            },
            "was_ray_initialized": False,
            "verbose": False,
            "seed": 0,
            "columns": ["feature_0", "feature_1", "feature_2"],
        }

        assert type(discretizer) == KmedoidDiscritizer
        assert str(discretizer.__dict__) == str(expected_discretizer__dict__)

    def test_transform(self):
        """Test transform function"""
        X = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        y = pd.DataFrame.from_dict({f"feature_{i}": [1, 1, 0] for i in range(3)})
        discretizer = KmedoidDiscritizer()
        discretizer = discretizer.fit(X)
        y_transform = discretizer.transform(y)

        assert y_transform.to_dict() == {
            "feature_0": {0: 0, 1: 0, 2: 0},
            "feature_1": {0: 0, 1: 0, 2: 0},
            "feature_2": {0: 0, 1: 0, 2: 0},
        }

    def test_fit_transform(self):
        """Test fit_transform function"""
        X = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        discretizer = KmedoidDiscritizer()
        x_transform = discretizer.fit_transform(X)

        assert x_transform.to_dict() == {
            "feature_0": {0: 0, 1: 0, 2: 1},
            "feature_1": {0: 0, 1: 0, 2: 1},
            "feature_2": {0: 0, 1: 0, 2: 1},
        }

    def test_sklearn_pipeline(self):
        """Test sklearn end to end pipeline"""
        X = pd.DataFrame.from_dict({f"feature_{i}": [1, 2, 3] for i in range(3)})
        y = pd.DataFrame.from_dict({f"feature_{i}": [1, 1, 0] for i in range(3)})
        num_features = X.columns.to_list()
        numeric_transformer = Pipeline([("discretizer", KmedoidDiscritizer())])
        preprocessor = ColumnTransformer([("num", numeric_transformer, num_features)])
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=1, random_state=0)),
            ]
        )
        clf.fit(X, y)

        assert clf.score(X, y) == 1.0


if __name__ == "__main__":
    unittest.main()
