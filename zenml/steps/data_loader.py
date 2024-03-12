import os
import pandas as pd
from zenml import step
from fastai.vision.all import DataBlock, ImageBlock, CategoryBlock, RandomSplitter, ColReader, Resize, IntToFloatTensor, Normalize
from typing_extensions import Annotated
from torchvision.transforms import CenterCrop

@step
def create_dataloader(data: pd.DataFrame, random_state: int=42) ->Annotated[DataBlock, "Dataloader"]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    batch_tfms = [IntToFloatTensor(), Normalize(mean=mean, std=std)]
    dl = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=RandomSplitter(seed=random_state),
                   get_x=ColReader('path'),
                   get_y=ColReader('class_label'),
                   item_tfms=[Resize(256), CenterCrop(256)],
                   batch_tfms=batch_tfms)
    dls = dl.dataloaders(data)
    return dls