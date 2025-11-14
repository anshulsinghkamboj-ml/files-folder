import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("randomforest_Experiment")


from logging_config import get_logger

logger=get_logger(__name__)

logger.info("loading data")
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame

# 2. Clean missing values ("?" → NaN)
df = df.replace("?", pd.NA)

logger.info("x,y split")
# 3. Split X and y
X = df.drop("class", axis=1)
y = df["class"]

categorical_cols = X.select_dtypes(include="category").columns.tolist()
numerical_cols = X.select_dtypes(include="number").columns.tolist()

logger.info("using column transformer")
logger.info("using simple imputer for num cols")
logger.info("using ohe for categ cols")
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)
with mlflow.start_run() as run :
    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    logger.info("train test  split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("training model")
    model.fit(X_train, y_train)

    logger.info("saving  model")
    joblib.dump(model,'randomforest_model.pkl')


    from sklearn.metrics import accuracy_score

    preds = model.predict(X_test)
    score=accuracy_score(y_test, preds)

    logger.info(f"model accuracy score : {score}")

    from sklearn.metrics import roc_auc_score, average_precision_score

    y_proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba,pos_label=">50K")


    logger.info(f"model ROC-AUC : {roc}")
    logger.info(f"model PR-AUC : {pr}")

    mlflow.log_metric("accuracy", score)
    mlflow.log_metric("ROC-AUC", roc)
    mlflow.log_metric("PR-AUC", pr)

    signature=mlflow.models.infer_signature(X_train.sample(5, random_state=42), model.predict(X_train.sample(5, random_state=42)))
    input_example=X_test.iloc[[0]]


    mlflow.sklearn.log_model(
        sk_model=model,
        name="randomforest",
        signature=signature,
        input_example=input_example,
        registered_model_name="random forest",
    )