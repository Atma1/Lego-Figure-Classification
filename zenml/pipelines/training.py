from zenml import pipeline
from pipelines import data_engineering
from steps import train, model_promoter


@pipeline(enable_cache=False)
def training(lr: float):

    transformed_data = data_engineering()
    trained_model = train(transformed_data=transformed_data, lr=lr)

    model_promoter(trained_model=trained_model)