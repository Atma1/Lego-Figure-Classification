from zenml import step, ArtifactConfig
from fastai.vision.all import *
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def train(dls: DataLoaders, lr: float=1e-2
                ) -> Annotated[Learner, ArtifactConfig(name="resnet18", is_model_artifact=True)]:
    """
    Model promoter step
    """
    model = vision_learner(dls, resnet18, metrics=accuracy, lr=lr)
    logger.info("Training model.....")
    model.fine_tune(1);
    return model