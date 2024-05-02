from zenml import pipeline, log_artifact_metadata
from pipelines import data_engineering
from steps import train, model_promoter


@pipeline
def training(lr: float):

    dataloaders = data_engineering()
    trained_model = train(dls=dataloaders, lr=lr)
    
    metrics = trained_model.recorder.metrics
    accuracy = metrics[0].value.item()

    log_artifact_metadata(
        metadata={"accuracy": float(accuracy)},
        artifact_name="resnet18",
    )

    model_promoter(accuracy=accuracy)