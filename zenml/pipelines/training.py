from zenml import pipeline
from pipelines import data_engineering
from steps import train


@pipeline
def training_model(lr: float):
    dataloaders = data_engineering()
    trained_model = train(dls=dataloaders, lr=lr)
    error_rate = trained_model.recorder.values[-1][trained_model.recorder.metric_names.index('error_rate')]