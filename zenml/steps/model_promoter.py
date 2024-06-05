from zenml import get_step_context, step, log_artifact_metadata
from zenml.client import Client
from zenml.logger import get_logger
from fastai.vision.all import Learner

logger = get_logger(__name__)

@step
def model_promoter(trained_model: Learner, stage: str = "production") -> None:
    """
    Model promoter step
    """
    metrics = trained_model.recorder.metrics
    accuracy = metrics[0].value.item()

    log_artifact_metadata(
        metadata={"accuracy": float(accuracy)},
        artifact_name="resnet18",
    )

    if accuracy < 0.8:
        logger.info(
            f"Model accuracy {accuracy*100:.2f}% is below 80%! Not promoting model."
        )
    else:
        logger.info(f"Model promoted to {stage}!")

        current_model = get_step_context().model

        client = Client()
        
        try:
            stage_model = client.get_model_version(
                current_model.name, stage
            )
            prod_accuracy = (
                stage_model.get_artifact("resnet18")
                .run_metadata["accuracy"]
                .value
            )
            if float(accuracy) > float(prod_accuracy):
                current_model.set_stage(stage, force=True)
        except KeyError:
            current_model.set_stage(stage, force=True)