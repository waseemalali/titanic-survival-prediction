"""Titanic survival prediction - EVERYTHING IN ONE FILE.

How to run (put this file and train.csv in the same folder):

    pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
    python titanic_all_in_one.py

What it does, in order:
  1. loads train.csv and inspects it
  2. splits it 60% train / 20% validation / 20% test
  3. draws exploratory charts            -> outputs/figures/
  4. builds the 11-step cleaning pipeline and compares 4 models
  5. tunes the best one (KNN), saves it   -> outputs/models/
  6. evaluates ONCE on the untouched test set (expected ~0.849)
  7. predicts a few example passengers (you can also pass your own):

    python titanic_all_in_one.py --name "Smith, Mrs. Anna" --sex female --pclass 1 --age 30 --fare 80
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:          # no pop-up windows unless you ask for them
    matplotlib.use("Agg")

# --------------------------------------------------------------------------
# Paths and small helpers (these used to live in src/common.py)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
for _d in (FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
RESULTS = {}
SHOW = "--show" in sys.argv


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(name, fig=None):
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"   [figure saved] {path.relative_to(ROOT)}")


def log_result(stage, key, value):
    RESULTS.setdefault(stage, {})[key] = value


def write_results():
    path = OUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults written to {path.relative_to(ROOT)}")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# The Titanic project itself (this used to be src/titanic_pipeline.py)
# --------------------------------------------------------------------------
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier


STAGE = "titanic"

# --------------------------------------------------------------------------
# Custom transformers (each does ONE job; fit() learns, transform() applies)
# --------------------------------------------------------------------------
TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Mlle": "Miss",
    "Ms": "Mrs", "Mme": "Mrs", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare",
}


class TypeConverter(BaseEstimator, TransformerMixin):
    """Force each column to the dtype we expect."""
    def __init__(self, type_mapping):
        self.type_mapping = type_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, dtype in self.type_mapping.items():
            X[col] = X[col].astype(dtype)
        return X


class SexMapping(BaseEstimator, TransformerMixin):
    """Unify spellings of the Sex column (male/M -> Male, female/F -> Female)."""
    def __init__(self, sex_col="Sex"):
        self.sex_col = sex_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mapping = {"male": "Male", "female": "Female", "M": "Male", "F": "Female"}
        X[self.sex_col] = X[self.sex_col].replace(mapping)
        return X


class SetIndexByPassengerId(BaseEstimator, TransformerMixin):
    """PassengerId is an identifier, not a feature -> move it to the index."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.set_index("PassengerId")


class ColumnImputer(BaseEstimator, TransformerMixin):
    """Fill missing values of ONE column. The fill value is learned on the
    training data only (this is what prevents data leakage)."""
    def __init__(self, strategy="mean", column_name=None):
        self.strategy = strategy
        self.column_name = column_name

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X[[self.column_name]])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column_name] = self.imputer_.transform(X[[self.column_name]]).ravel()
        return X


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Create FamilySize and Title (extracted from the passenger name)."""
    def __init__(self, add_title=True, add_family_size=True):
        self.add_title = add_title
        self.add_family_size = add_family_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_family_size:
            X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        if self.add_title:
            title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
            X["Title"] = title.map(TITLE_MAP).fillna("Rare")
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are not useful as model input."""
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=list(self.features_to_drop or []))


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """One-hot encode the given columns and return a DataFrame."""
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(X[self.column])
        return self

    def transform(self, X):
        X = X.copy()
        enc = self.encoder_.transform(X[self.column])
        enc_df = pd.DataFrame(enc, index=X.index,
                              columns=self.encoder_.get_feature_names_out(self.column))
        return pd.concat([X.drop(columns=self.column), enc_df], axis=1)


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """Turn one text column into integer codes."""
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        self.encoder_ = LabelEncoder().fit(X[self.column])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = self.encoder_.transform(X[self.column])
        return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    """StandardScaler on some columns, MinMaxScaler on others."""
    def __init__(self, standard_scaler_features=None, min_max_scaler_features=None):
        self.standard_scaler_features = standard_scaler_features
        self.min_max_scaler_features = min_max_scaler_features

    def fit(self, X, y=None):
        self.std_ = StandardScaler().fit(X[self.standard_scaler_features])
        self.mm_ = MinMaxScaler().fit(X[self.min_max_scaler_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.standard_scaler_features] = self.std_.transform(X[self.standard_scaler_features])
        X[self.min_max_scaler_features] = self.mm_.transform(X[self.min_max_scaler_features])
        return X


# --------------------------------------------------------------------------
# Config + pipeline factory
# --------------------------------------------------------------------------
TYPE_MAPPING = {
    "PassengerId": int, "Pclass": int, "Name": object, "Sex": object,
    "Age": float, "SibSp": int, "Parch": int, "Ticket": object,
    "Fare": float, "Cabin": object, "Embarked": object,
}
FEATURES_TO_DROP = ["Name", "SibSp", "Parch", "Cabin", "Ticket"]
STANDARD_SCALED = ["Pclass", "Age", "Sex", "FamilySize", "Title",
                   "Embarked_Q", "Embarked_S", "Embarked_C"]
MINMAX_SCALED = ["Fare"]


def build_preprocessing():
    """The whole cleaning + feature-engineering chain as ONE sklearn Pipeline."""
    return Pipeline(steps=[
        ("type_converter", TypeConverter(TYPE_MAPPING)),
        ("sex_mapping", SexMapping("Sex")),
        ("set_index", SetIndexByPassengerId()),
        ("age_imputer", ColumnImputer("median", "Age")),
        ("embarked_imputer", ColumnImputer("most_frequent", "Embarked")),
        ("feature_engineering", FeatureEngineering(True, True)),
        ("drop_features", DropColumns(FEATURES_TO_DROP)),
        ("onehot_embarked", OneHotEncoderTransformer(["Embarked"])),
        ("label_sex", LabelEncoderTransformer("Sex")),
        ("label_title", LabelEncoderTransformer("Title")),
        ("scaler", ScalerTransformer(STANDARD_SCALED, MINMAX_SCALED)),
    ])


# --------------------------------------------------------------------------
# Stage steps
# --------------------------------------------------------------------------
def load_and_inspect():
    header("5.1  Load and inspect the data")
    data = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Shape: {data.shape}")
    print("\nMissing values per column:")
    print(data.isnull().sum()[lambda s: s > 0])
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print(f"Overall survival rate: {data['Survived'].mean():.3f}")
    log_result(STAGE, "rows", int(len(data)))
    log_result(STAGE, "columns", int(data.shape[1]))
    log_result(STAGE, "missing", {k: int(v) for k, v in data.isnull().sum().items() if v})
    log_result(STAGE, "survival_rate", round(float(data["Survived"].mean()), 4))
    return data


def split_data(data):
    header("5.2  Stratified split: 60% train / 20% validation / 20% test")
    X, y = data.drop("Survived", axis=1), data["Survived"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    for name, part in (("train", y_train), ("validation", y_val), ("test", y_test)):
        print(f"  {name:<11}: {len(part):>4} rows, survival rate {part.mean():.3f}")
    log_result(STAGE, "split_sizes", {"train": len(y_train), "val": len(y_val),
                                      "test": len(y_test)})
    return X_train, X_val, X_test, y_train, y_val, y_test


def explore(X_train, y_train, full_data):
    """EDA on the TRAINING set only (the test set must stay untouched)."""
    header("5.3  Exploratory data analysis (figures)")
    train = pd.concat([X_train, y_train], axis=1)

    plt.figure(figsize=(5, 4))
    sns.countplot(x=y_train, hue=y_train, legend=False)
    plt.title("Count of survivors (0 = no, 1 = yes)")
    save_fig("titanic_01_target")

    plt.figure(figsize=(7, 4))
    sns.heatmap(X_train.isnull(), cbar=False, cmap="binary")
    plt.title("Missing values (black = missing)")
    save_fig("titanic_02_missing")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x="Sex", y="Survived", data=train, hue="Sex", ax=axes[0])
    axes[0].set_title("Survival rate by sex")
    sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train, ax=axes[1])
    axes[1].set_title("Survival rate by class and sex")
    save_fig("titanic_03_survival_rates", fig)

    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=train, x="Age", hue="Survived", bins=30, kde=True)
    plt.title("Age distribution by survival status")
    save_fig("titanic_04_age")

    num = train.select_dtypes(include="number").drop(columns=["PassengerId"])
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation of numeric features")
    save_fig("titanic_05_correlation")

    # The catplot exercise from the notebook, now built from train.csv itself
    # (the notebook loaded seaborn's online copy, which fails without internet)
    sns.set_theme(style="whitegrid")
    plot_df = full_data.assign(Survived=full_data["Survived"].map({0: "No", 1: "Yes"}))
    g = sns.catplot(data=plot_df, x="Embarked", hue="Survived", col="Pclass", kind="count",
                    order=["S", "C", "Q"], hue_order=["No", "Yes"],
                    palette={"No": "#d62728", "Yes": "#2ca02c"}, height=4.5, aspect=0.9)
    g.set_axis_labels("Port of Embarkation", "Number of Passengers")
    g.set_titles("Pclass {col_name}")
    g.figure.suptitle("Titanic Passengers: Embarkation Port by Survival Status and Class",
                      y=1.05, fontsize=14, fontweight="bold")
    save_fig("titanic_06_catplot", g.figure)
    sns.set_theme()  # back to defaults for later figures


def baseline_and_cv(X_train, y_train):
    header("5.4  Baseline and leak-free cross-validation")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy = Pipeline([("prep", build_preprocessing()),
                      ("model", DummyClassifier(strategy="most_frequent"))])
    base = cross_val_score(dummy, X_train, y_train, cv=skf).mean()
    print(f"Baseline (always predict 'did not survive'): {base:.4f}")
    print("Because the preprocessing lives INSIDE the pipeline, every CV fold "
          "learns its own medians/encoders -> no data leakage.")
    log_result(STAGE, "baseline_cv_accuracy", round(float(base), 4))
    return base


def compare_models(X_train, y_train, X_val, y_val):
    header("5.5  Compare algorithms on the validation set")
    prep = build_preprocessing()
    X_tr = prep.fit(X_train).transform(X_train)
    X_va = prep.transform(X_val)
    print(f"  Features after the pipeline: {list(X_tr.columns)}")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_train)
        tr = accuracy_score(y_train, model.predict(X_tr))
        va = accuracy_score(y_val, model.predict(X_va))
        rows.append({"model": name, "train_acc": round(tr, 4), "val_acc": round(va, 4)})
        print(f"  {name:<20} train={tr:.4f}  validation={va:.4f}")
    log_result(STAGE, "model_comparison", rows)
    print("\n  Note: Decision Tree / Random Forest score ~0.98 on train but far lower "
          "on validation -> over-fitting.")

    knn = models["KNN"]
    pred = knn.predict(X_va)
    cm = confusion_matrix(y_val, pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not survived", "Survived"],
                yticklabels=["Not survived", "Survived"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix - KNN (validation)")
    save_fig("titanic_07_confusion_knn")
    print("\n" + classification_report(y_val, pred))

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values()
    plt.figure(figsize=(6.5, 4.5))
    imp.plot.barh(color="#4c72b0")
    plt.title("What matters most? (Random Forest feature importance)")
    save_fig("titanic_08_feature_importance")
    log_result(STAGE, "feature_importance", {k: round(float(v), 4) for k, v in imp[::-1].items()})
    return rows


def tune_and_finalize(X_train, y_train, X_val, y_val, X_test, y_test, quick):
    header("5.6  Fine-tune KNN inside the full pipeline")
    full = Pipeline([("preprocessing", build_preprocessing()),
                     ("model", KNeighborsClassifier())])
    full.fit(X_train, y_train)
    print(f"Untuned pipeline  - validation accuracy: "
          f"{accuracy_score(y_val, full.predict(X_val)):.4f}")

    # n_jobs=1 on purpose: the dataset is tiny and multiprocessing on Windows
    # can hang (the original notebook's grid search had to be interrupted).
    grid = GridSearchCV(full, {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }, cv=5, scoring="accuracy", n_jobs=1)
    grid.fit(X_train, y_train)
    print(f"Grid search       - best CV {grid.best_score_:.4f}  params {grid.best_params_}")
    print(f"                    validation accuracy: "
          f"{accuracy_score(y_val, grid.best_estimator_.predict(X_val)):.4f}")

    rand = RandomizedSearchCV(full, {
        "model__n_neighbors": np.arange(1, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "chebyshev"],
    }, n_iter=20, cv=5, scoring="accuracy", n_jobs=1, random_state=SEED)
    rand.fit(X_train, y_train)
    best = rand.best_estimator_
    val_acc = accuracy_score(y_val, best.predict(X_val))
    clean = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in rand.best_params_.items()}
    print(f"Randomized search - best CV {rand.best_score_:.4f}  params {clean}")
    print(f"                    validation accuracy: {val_acc:.4f}")

    header("5.7  Save the model, reload it, and evaluate ONCE on the test set")
    path = MODEL_DIR / "best_titanic_pipeline.joblib"
    joblib.dump(best, path)
    print(f"Saved -> {path}")
    loaded = joblib.load(path)
    test_acc = accuracy_score(y_test, loaded.predict(X_test))
    print(f"FINAL test accuracy (data never seen before): {test_acc:.4f}")

    log_result(STAGE, "grid_best_params", grid.best_params_)
    log_result(STAGE, "grid_best_cv", round(float(grid.best_score_), 4))
    log_result(STAGE, "random_best_params", {k: (int(v) if isinstance(v, np.integer) else v)
                                             for k, v in rand.best_params_.items()})
    log_result(STAGE, "random_best_cv", round(float(rand.best_score_), 4))
    log_result(STAGE, "validation_accuracy_final", round(float(val_acc), 4))
    log_result(STAGE, "test_accuracy_final", round(float(test_acc), 4))
    return loaded


def predict_demo(model, X_test, y_test):
    """Show the saved model predicting a few unseen passengers."""
    header("5.8  Prediction demo on 5 unseen passengers")
    sample = X_test.head(5)
    proba = model.predict_proba(sample)[:, 1]
    pred = model.predict(sample)
    for (_, row), p, yhat, truth in zip(sample.iterrows(), proba, pred, y_test.head(5)):
        print(f"  {row['Name'][:32]:<32} class {row['Pclass']}  {row['Sex']:<6} "
              f"P(survive)={p:.2f}  predicted={yhat}  actual={truth}")


def run(quick=True):
    data = load_and_inspect()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    explore(X_train, y_train, data)
    baseline_and_cv(X_train, y_train)
    compare_models(X_train, y_train, X_val, y_val)
    model = tune_and_finalize(X_train, y_train, X_val, y_val, X_test, y_test, quick)
    predict_demo(model, X_test, y_test)


# --------------------------------------------------------------------------
# Live prediction (this used to be predict.py)
# --------------------------------------------------------------------------
EXAMPLES = [
    dict(Name="Example, Mrs. Rose", Sex="female", Pclass=1, Age=28, SibSp=1, Parch=0, Fare=90.0, Embarked="C"),
    dict(Name="Example, Mr. Jack", Sex="male", Pclass=3, Age=20, SibSp=0, Parch=0, Fare=7.9, Embarked="S"),
    dict(Name="Example, Master. Tom", Sex="male", Pclass=2, Age=6, SibSp=1, Parch=1, Fare=30.0, Embarked="S"),
    dict(Name="Example, Miss. Mary", Sex="female", Pclass=3, Age=22, SibSp=0, Parch=0, Fare=8.0, Embarked="Q"),
]


def to_frame(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "PassengerId", range(10001, 10001 + len(df)))
    df["Ticket"], df["Cabin"] = "N/A", None
    return df[["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
               "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]


def predict_passengers(model, rows):
    header("6  Live prediction")
    proba = model.predict_proba(to_frame(rows))[:, 1]
    print(f"{'Passenger':<24}{'Class':<7}{'Sex':<8}{'Age':<6}{'P(survive)':<12}Prediction")
    print("-" * 70)
    for r, pr in zip(rows, proba):
        verdict = "SURVIVES" if pr >= 0.5 else "does not survive"
        print(f"{r['Name'][:23]:<24}{r['Pclass']:<7}{r['Sex']:<8}{r['Age']:<6.0f}{pr:<12.2f}{verdict}")


def main():
    p = argparse.ArgumentParser(description="Titanic: train, evaluate, save and predict - one file.")
    p.add_argument("--show", action="store_true", help="show figures in windows too")
    p.add_argument("--name", help='e.g. "Smith, Mrs. Anna"  (format: Surname, Title. First)')
    p.add_argument("--sex", choices=["male", "female"])
    p.add_argument("--pclass", type=int, choices=[1, 2, 3])
    p.add_argument("--age", type=float)
    p.add_argument("--sibsp", type=int, default=0)
    p.add_argument("--parch", type=int, default=0)
    p.add_argument("--fare", type=float)
    p.add_argument("--embarked", choices=["S", "C", "Q"], default="S")
    a = p.parse_args()

    if a.name:
        missing = [k for k in ("sex", "pclass", "age", "fare") if getattr(a, k) is None]
        if missing:
            sys.exit("Please also give: " + ", ".join("--" + m for m in missing))
        rows = [dict(Name=a.name, Sex=a.sex, Pclass=a.pclass, Age=a.age, SibSp=a.sibsp,
                     Parch=a.parch, Fare=a.fare, Embarked=a.embarked)]
    else:
        rows = EXAMPLES

    if not (DATA_DIR / "train.csv").exists():
        sys.exit("train.csv not found. Put it in the same folder as this file.")

    t0 = time.time()
    try:
        run(quick=True)
    except Exception:
        traceback.print_exc()
        sys.exit("Training failed - see the error above.")
    model = joblib.load(MODEL_DIR / "best_titanic_pipeline.joblib")
    predict_passengers(model, rows)
    write_results()
    print(f"\nDONE in {time.time() - t0:.0f}s   |   figures: {FIG_DIR}   |   model: {MODEL_DIR}")


if __name__ == "__main__":
    main()
