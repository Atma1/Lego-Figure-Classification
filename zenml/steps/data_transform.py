import pandas as pd
import os
from zenml import step
from typing_extensions import Annotated

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

@step
def data_transform(df_dir: str) -> Annotated[pd.DataFrame, "transformed_data"]:
    metadata_csv_dir = os.path.join(ROOT_DIR , f"{df_dir}/metadata.csv")
    index_csv_dir = os.path.join(ROOT_DIR, f"{df_dir}/index.csv")
    index_label_series = pd.read_csv(metadata_csv_dir)["minifigure_name"]
    index_csv_df = pd.read_csv(index_csv_dir)
    index_csv_df["path"] = index_csv_df["path"].map(lambda value: f"{ROOT_DIR}/{value}")
    index_csv_df["class_id"] = index_csv_df["class_id"].map(lambda value: index_label_series[int(value)-1])
    transformed_data = index_csv_df.rename({"path": "path", "class_id": "class_label"})
    return transformed_data



