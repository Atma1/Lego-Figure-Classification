import os
import pandas as pd
from zenml import step
from fastai.vision.all import *
from typing_extensions import Annotated
# from torchvision.transforms import CenterCrop

@step
def data_loader(data: pd.DataFrame, random_state: int=42, valid_pct: float=0.2
                      ) ->Annotated[DataLoaders, "Dataloader"]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    batch_tfms = [IntToFloatTensor, Normalize(mean=mean, std=std)]
    dl = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=RandomSplitter(seed=random_state, valid_pct=valid_pct),
                   get_x=ColReader('path'),
                   get_y=ColReader('class_label'),
                   item_tfms=[ToTensor, Resize(256)],
                   batch_tfms=aug_transforms(size=256))
    dls = dl.dataloaders(data)
    return dls