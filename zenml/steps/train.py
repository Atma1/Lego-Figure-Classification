from zenml import step, ArtifactConfig
from fastai.vision.all import *
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)

def data_loader(data: pd.DataFrame, random_state: int=42, valid_pct: float=0.2
                      ) ->Annotated[DataLoaders, "Dataloader"]:
    """
    Data loader step
    """
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    batch_tfms = [Normalize.from_stats(mean=mean, std=std)]
    dl = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=RandomSplitter(seed=random_state, valid_pct=valid_pct),
                   get_x=ColReader('img_path'),
                   get_y=ColReader('class_label'),
                   item_tfms=[CropPad(256), Resize(256)],
                   batch_tfms=batch_tfms)
    dls = dl.dataloaders(data, verbose=True)
    return dls

@step
def train(transformed_data: pd.DataFrame, lr: float=1e-2
                ) -> Annotated[Learner, ArtifactConfig(name="resnet18", is_model_artifact=True)]:
    """
    Model promoter step
    """
    dls = data_loader(transformed_data)
    learn = vision_learner(dls, resnet18, metrics=accuracy, lr=lr)
    logger.info("Training model.....")
    learn.fine_tune(10)
    return learn