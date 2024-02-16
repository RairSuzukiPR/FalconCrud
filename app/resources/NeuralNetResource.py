import random

import falcon
import json
import torch
import numpy as np
# from falcon.media.validators.jsonschema import validate
# from app.schemas import load_schema
from app.services.model import Linear_QNet, QTrainer
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class ModelTrainerAgentManager:
    def __init__(self):
        self.n_games = 0
        self.wins = 0
        # self.epsilon = 0  # randomness
        self.gamma = 0  # discount rate
        self.learning_rate = 0
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.neural_net = None
        self.trainer = None

    def initialize_neural_net(self, input_size, hidden_size, output_size, learning_rate, gamma):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.neural_net = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.neural_net, lr=learning_rate, gamma=gamma)

    def predict(self, state):
        # print(state)
        if self.neural_net and self.trainer:
            self.neural_net.eval()
            final_move = [0, 0, 0, 0, 0]
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.neural_net(state0)
            # print('-----start-----')
            # print('prediction -> ', prediction)
            move = torch.argmax(prediction).item()
            # print('move -> ', move)
            # print('return -> ', int(np.array(move).argmax()))
            # print('-----end-----')
            final_move[move] = 1
            # return move
            return int(np.array(final_move).argmax())
        return 0  # TODO: temp

    def train(self, state_old, action, reward, next_state, done):
        if self.neural_net and self.trainer:
            self.trainer.train_step(state_old, action, reward, next_state, done)
            self._remember(state_old, action, reward, next_state, done)

            if done:
                self.n_games += 1
                self._train_long_memory()
                self.neural_net.save()

                # TODO: entender
                # if score > record:
                #     record = score
                #     self.neural_net.save()

                # TODO: ver uma forma de plotar wins por game que nao usou acao radom, usar reward por round
                # plot_scores.append(score)
                # total_score += score
                # mean_score = total_score / agent.n_games
                # plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)
    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def _train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # print(states.shape)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



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
            res.get('hidden_size'),
            res.get('output_size'),
            res.get('learning_rate'),
            res.get('gamma')
        )

        resp.status = falcon.HTTP_200
        resp.text = json.dumps({'message': 'Neural net and trainer initialized successfully'})

    def on_patch(self, req, resp):
        res = req.get_media()
        manager.train(
            res.get('state'),
            res.get('action'),
            res.get('reward'),
            res.get('new_state'),
            res.get('done')
        )

        resp.status = falcon.HTTP_200


NeuralNetResource = NeuralNetResource()
