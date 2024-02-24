import falcon
import json
import torch
from app.services.model import NeuralNet, Trainer


class ModelTrainerAgentManager:
    def __init__(self):
        self.neural_net = None
        self.trainer = None

    def initialize_neural_net(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs):
        self.neural_net = NeuralNet(input_size, hidden1_size, hidden2_size, output_size)
        self.trainer = Trainer(self.neural_net, lr=learning_rate, epochs=epochs)

    def predict(self, state):
        if self.neural_net and self.trainer:
            new_state = torch.FloatTensor(state)
            with torch.no_grad():
                print("best action ->", self.trainer.model(new_state).argmax().item())
                return self.trainer.model(new_state).argmax().item()
            return -1

    def train(self):
        if self.neural_net and self.trainer:
            return self.trainer.train()


manager = ModelTrainerAgentManager()


class NeuralNetResource:
    def on_get(self, req, resp):
        res = req.get_param_as_list('state')
        if res:
            try:
                res = [float(num) for num in res]
                resp.status = falcon.HTTP_200
                resp.media = manager.predict(res)
            except ValueError:
                resp.media = {'error': 'One or more values are not valid'}
        else:
            resp.media = {'error': 'No numbers provided'}

    def on_post(self, req, resp):
        res = req.get_media()
        manager.initialize_neural_net(
            res.get('input_size'),
            res.get('hidden1_size'),
            res.get('hidden2_size'),
            res.get('output_size'),
            res.get('learning_rate'),
            res.get('epochs')
        )

        resp.status = falcon.HTTP_200
        resp.text = json.dumps({'message': 'Neural net and trainer initialized successfully'})

    def on_patch(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.text = json.dumps({'message': manager.train()})


NeuralNetResource = NeuralNetResource()
