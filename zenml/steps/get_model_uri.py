from zenml import step
from zenml.logger import get_logger
from zenml.client import Client
from mlflow.tracking import artifact_utils
from typing_extensions import Annotated

logger = get_logger(__name__)

@step
def get_model_uri(run_id: str) -> Annotated[str, "model_uri"]:
    zenml_client = Client()
    experiment_tracker = zenml_client.active_stack.experiment_tracker
    experiment_tracker.configure_mlflow()
    model_name = "model"
    model_uri = artifact_utils.get_artifact_uri(
        run_id=run_id, artifact_path=model_name
    )
    return model_uri
