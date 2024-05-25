from zenml import pipeline, Model
from pipelines import data_engineering
from steps import train, model_promoter


model = Model(
    name="Resnet18",
    license="MIT",
    description="Resnet18 feature extraction model for classification of lego figure.",
    version="stagin"
)

@pipeline(enable_cache=True, model=model)
def training(lr: float):

    transformed_data = data_engineering()
    trained_model = train(transformed_data=transformed_data, lr=lr)

    model_promoter(trained_model=trained_model)