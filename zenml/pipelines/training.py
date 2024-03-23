from zenml import pipeline
from pipelines import data_engineering
from steps import train


@pipeline
def training_model(lr: float):
    dataloaders = data_engineering()
    trained_model = train(dls=dataloaders, lr=lr)
    metrics = trained_model.recorder.metrics
    accuracy = metrics[0].value.item()