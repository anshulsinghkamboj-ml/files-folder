from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_handler import load_process_pipeline
from utils.performance_matrix import model_preformance

import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment("randomforest")
client = MlflowClient()

with mlflow.start_run():

    preprocess, X, y = load_process_pipeline()

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # correct unpacking order
    accuracy, roc, pr, tn, fp, fn, tp = model_preformance(
        model, X_test, y_test, preds
    )

    # log params
    mlflow.log_params(model.named_steps["clf"].get_params())

    # log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr)
    mlflow.log_metric("TN", tn)
    mlflow.log_metric("TP", tp)
    mlflow.log_metric("FN", fn)
    mlflow.log_metric("FP", fp)

    # log the model
    mlflow.sklearn.log_model(model, name="model")

    # get run id
    run_id = mlflow.active_run().info.run_id
    print("RUN ID:", run_id)

# ----------------------------
# MODEL REGISTRY (NO STAGES)
# ----------------------------

model_uri = f"runs:/{run_id}/model"

# create registered model (ignore if exists)
try:
    client.create_registered_model("adult_random_forest")
except:
    pass

# register a new model version
mv = client.create_model_version(
    name="adult_random_forest",
    source=model_uri,
    run_id=run_id
)

version_number = mv.version
print(f"Registered new model version: {version_number}")

print("\nNext step: Load this version in FastAPI using:")
print(f"mlflow.pyfunc.load_model('models:/adult_random_forest/{version_number}')")
