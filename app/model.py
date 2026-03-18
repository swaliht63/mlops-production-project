import mlflow.pyfunc

def load_model():
    return mlflow.pyfunc.load_model("models:/iris-model/1")