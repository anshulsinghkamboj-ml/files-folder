# mlflow_loader.py
from mlflow import MlflowClient
import mlflow

MLFLOW_URI = "sqlite:///mlflow.db"
MODEL_NAME = "adult_random_forest"

def load_latest_model():
    """
    Loads the latest version of a registered MLflow model.
    Returns an MLflow PyFunc model instance.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # search registered model versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"No versions found for registered model '{MODEL_NAME}'. "
            "Ensure trainer.py has registered at least one version."
        )

    # pick the latest version
    latest_version = max(int(v.version) for v in versions)
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"

    # load pyfunc model
    model = mlflow.pyfunc.load_model(model_uri)
    return model, latest_version
