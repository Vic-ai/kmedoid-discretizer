# kmedoid-discritizer
Adaptative Kmedoid discritizer for numerical feature engineering.

[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%5E0.24.2-blue)](https://github.com/scikit-learn/scikit-learn)
[![Python](https://img.shields.io/badge/python-%3E%3D%203.7.13%2C%20%3C%3D%203.9.16-blue)](https://www.python.org/downloads/release/python-3916/)
[![Test](https://github.com/Vic-ai/KmedoidDiscritizer/actions/workflows/.test.yml/badge.svg)](https://github.com/Vic-ai/KmedoidDiscritizer/actions/workflows/.test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
kmedoid-discritizer (Adaptative Kmedoid discritizer) allows to discritize numerical feature into `n_bins` using Kmedoids Clustering algrorithm compatible sklearn (Alternative to sklearn KBinsDiscretizer).
With this implemenation, we can have:
- A custom number of bins for each numeral feature. Kmedoids will be run for each columns.
- Adapt the number of bins dynamically whenever this one is two high (more precesly when two centroids are assigned to the same data point.)
- Multiple Backends are possible: serial, multiprocessing, and ray to speed up the Kmedoids compuation.
- Mainly use Pandas DataFrame and Numpy array.

## Install

```
pip install git+ssh://git@github.com/Vic-ai/kmedoid-discritizer.git
```

### Play with the code and run it locally without pip 
`git clone git@github.com:Vic-ai/kmedoid-discritizer.git `
- Download poetry (`make poetry-download`). See poetry doc: https://python-poetry.org/docs/
- Install the dev requirements into a virtualenv. (`make install`)

## Usage

### Basic Usage

Here is the Basic use-case data
``` python
# Fake training set
X = pd.DataFrame.from_dict({f"feature": [1, 2, 2, 3]})
# Fake Testing set
X_test = pd.DataFrame.from_dict({f"feature": [0, 2, 5]})
```

#### Ordinal encoding

```python
discritizer = KmedoidDiscritizer(2)
# discritize X into 2 bins => 1 and 2 will go in bin 0 and 3 in bin 1.
X_discrite = discritizer.fit_transform(X)
print(X_discrite)
# discritize X_test into 2 bins => 0 and 2 will go in bin 0 and 5 in bin 1.
X_test_discrite = discritizer.transform(X_test)
print(X_test_discrite)
```
```bash
   feature
0        0
1        0
2        0
3        1
   feature
0        0
1        0
2        1
```

### Onehot encoding
```python
discritizer = KmedoidDiscritizer(2, encoding="onehot-dense")
# discritize X into 2 bins => 1 and 2 will go in bin 0 and 3 in bin 1.
X_discrite = discritizer.fit_transform(X)
print(X_discrite)
# discritize X_test into 2 bins => 0 and 2 will go in bin 0 and 5 in bin 1.
X_test_discrite = discritizer.transform(X_test)
print(X_test_discrite)
```
```bash
   index    0    1
0      0  1.0  0.0
1      1  1.0  0.0
2      2  1.0  0.0
3      3  0.0  1.0
   index    0    1
0      0  1.0  0.0
1      1  1.0  0.0
2      2  0.0  1.0
```

### Advanced Usage Titanic (Sklearn Pipeline)

### Libraries
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kmedoid_discritizer.discritizer import KmedoidDiscritizer
from kmedoid_discritizer.utils.utils_external import PandasSimpleImputer

np.random.seed(0)
```

### Titanic Dataset
```python
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

cat_features = ["pclass", "sex"]
num_features = ["age", "fare", "sibsp", "parch"] # The one we will discritize

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```


### Training Pipeline
```python
# Numerical Transformer Pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", PandasSimpleImputer(strategy="median")),
        ("discritizer", KmedoidDiscritizer(
                            n_bins=[8, 5, 7, 7],
                            encode="onehot-dense",
                            backend="serial",
                            verbose=True,
                            seed=0,
                        )),
    ]
)

# Categorical Transformer Pipeline
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder()),
    ]
)

# The Combination of Numerical and Categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# Overall Pipeline preprocessor + classifier
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

clf.fit(X_train, y_train)
print("Train score: %.3f" % clf.score(X_train, y_train))
print("Test score: %.3f" % clf.score(X_test, y_test))
```

```bash
Train score: 0.802
Test score: 0.809
```

## Contributors
Marvin Martin

Daniel Nowak

## License
MIT License
Vic.ai 2023
