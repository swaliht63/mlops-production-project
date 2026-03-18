import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train():
    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)

        model = RandomForestClassifier()
        model.fit(X, y)

        mlflow.log_param("model", "RandomForest")

        # 👇 THIS IS IMPORTANT
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris-model"
        )

if __name__ == "__main__":
    train()