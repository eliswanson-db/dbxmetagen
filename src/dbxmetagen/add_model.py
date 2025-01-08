from mlflow.pyfunc import PythonModel
from mlflow.models import set_model

from src.dbxmetagen.calculator import add


class AddModel(PythonModel):
    def load_context(self, context):
        pass

    def predict(self, model_input, params=None):
        return add(model_input["x"], model_input["y"])


set_model(AddModel())