from zenml import step, ArtifactConfig
from fastai.vision.all import *
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def train_model(dls: DataLoaders, lr: float=1e-2
                ) -> Annotated[Learner, ArtifactConfig(name="Resnet18", is_model_artifact=True)]:
    model = vision_learner(dls, resnet18, metrics=mse, lr=lr)
    logger.info("Training model.....")
    model.fine_tune(1);
    return model