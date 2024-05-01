from zenml import pipeline
from zenml.logger import get_logger
from steps import (
    data_loader, 
    data_transform
)

logger = get_logger(__name__)

@pipeline
def data_engineering():
    """
    Data engineering
    """

    transformed_data = data_transform.data_transform("dataset")
    dls = data_loader.create_dataloader(transformed_data)
    return dls