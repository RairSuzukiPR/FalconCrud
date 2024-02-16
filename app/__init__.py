import falcon
from app.resources import NeuralNetResource
# from app.util_db import tables


app = falcon.App()

app.add_route('/init-neural-net', NeuralNetResource.NeuralNetResource)
app.add_route('/predict', NeuralNetResource.NeuralNetResource)
app.add_route('/train', NeuralNetResource.NeuralNetResource)
