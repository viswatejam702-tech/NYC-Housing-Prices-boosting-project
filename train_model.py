import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError(
        "CatBoost is required for this script. Install it with 'pip install catboost'."
    ) from exc

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    r"C:\ds and AI\ALL DATASETS\nyc_housing_base.csv"
)

# =====================================================
# TARGET
# =====================================================

TARGET = "sale_price"

# =====================================================
# FEATURE SELECTION
# =====================================================

features = [

    "zip_code",

    "yearbuilt",

    "lotarea",
    "bldgarea",
    "resarea",
    "comarea",

    "unitsres",
    "unitstotal",

    "numfloors",

    "latitude",
    "longitude",

    "building_age"

]

X = df[features]
y = df[TARGET]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# PREPROCESSING
# =====================================================

num_cols = X.columns

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    )
])

# =====================================================
# MODEL
# =====================================================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    verbose=100
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# =====================================================
# TRAIN
# =====================================================

pipe.fit(X_train, y_train)

# =====================================================
# EVALUATE
# =====================================================

pred = pipe.predict(X_test)

r2 = r2_score(y_test, pred)

mae = mean_absolute_error(y_test, pred)

print("\nR2 Score:", r2)
print("MAE:", mae)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    pipe,
    "nyc_housing_ai.pkl"
)

print("\nModel Saved Successfully")