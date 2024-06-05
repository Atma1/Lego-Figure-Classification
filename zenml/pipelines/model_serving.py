from zenml import pipeline, Model
from pipelines import data_engineering
from steps import train, model_promoter, serve_model, get_model_uri

model = Model(
    name="Resnet18",
    license="MIT",
    description="Resnet18 feature extraction model for classification of lego figure.",
    version="stagin"
)

@pipeline(model=model)
def model_serving(lr: float):
    transformed_data = data_engineering()
    trained_model, run_id = train(transformed_data=transformed_data, lr=lr)
    model_promoter(trained_model=trained_model)
    model_uri = get_model_uri(run_id)
    serve_model(model_uri)

if __name__ == '__main__':
    model_serving(lr=1e-3)