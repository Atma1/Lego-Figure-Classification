from zenml import pipeline
from zenml.logger import get_logger
from steps import (
    data_transform
)
from typing_extensions import Annotated
import pandas as pd

logger = get_logger(__name__)

@pipeline
def data_engineering() ->Annotated[pd.DataFrame, "transformed_data"]:
    """
    Data engineering
    """

    transformed_data = data_transform("dataset")
    return transformed_data