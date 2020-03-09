from Agents.BaseAgent import BaseAgent
import numpy as np
import math

class TTDAgent(BaseAgent):
    def __init__(self, actions, feature_name, gamma= 0.9, lamda= 0.5 , step_size = 0.1, num_state= 19, aggregation= 3, num_tiling= 3, tiles= 4, seed= 1):
        self.gamma = gamma
        self.lamda = lamda
        self.actions = actions
        self.step_size = step_size

        self.feature_name = feature_name
        self.num_states = num_state
        self.aggregation = aggregation
        self.num_tiling = num_tiling
        self.tiles = tiles
        # self.w = np.zeros(self.get_feature_vector_size())
        np.random.seed(seed)
        self.w = np.random.rand(self.get_feature_vector_size())
    def start(self, observation):
        self.prevX = self.feature_vector(observation)
        self.z = np.zeros_like(self.prevX)
        self.Vold = 0
        return self.policy(observation)

    def step(self, reward, observation):

        g = self.gamma
        l = self.lamda
        a = self.step_size
        self.x = self.feature_vector(observation)
        V = self.w.T.dot(self.prevX)
        Vprime = self.w.T.dot(self.x)
        tderror = reward + g * Vprime - V
        self.z = g * l * self.z + (1 - a * g * l * self.z.T.dot(self.prevX)) * self.prevX
        self.w += a * (tderror + V - self.Vold) * self.z - a * (V - self.Vold) * self.prevX
        self.Vold = Vprime
        self.prevX = np.copy(self.x)

        return self.policy(observation)

    def end(self, reward):
        g = self.gamma
        l = self.lamda
        a = self.step_size

        V = self.w.T.dot(self.prevX)
        Vprime = 0
        tderror = reward + g * Vprime - V
        self.z = g * l * self.z + (1 - a * g * l * self.z.T.dot(self.prevX)) * self.prevX
        self.w += a * (tderror + V - self.Vold) * self.z - a * (V - self.Vold) * self.prevX
        self.Vold = Vprime

    def policy(self, obs):
        # random policy
        return np.random.choice(self.actions)

    def feature_vector(self, obs):
        if self.feature_name == "onehot":
            # tabular (one-hot)
            a = np.zeros(self.num_states)
            a[obs] = 1
        elif self.feature_name == "binary":
            # binary
            a = bin(obs)[2:]
            dif = len(bin(self.num_states)[2:]) - len(a)
            a = list('0' * dif + a)
            for i in range(len(a)):
                a[i] = float(a[i])
            a = np.asarray(a)
        elif self.feature_name == "saggregation":
            a = np.zeros(math.ceil(self.num_states / self.aggregation))
            f = obs // self.aggregation
            a[f] = 1
            a = np.asarray(a)
        elif self.feature_name == "tilecoding":
            n = math.ceil(self.num_states / self.tiles)
            a = np.zeros(self.tiles * self.num_tiling)

            for t in range(self.num_tiling):
                copy_obs = obs - t
                f = copy_obs // n
                if f >= 0:
                    a[f + (t * self.tiles)] = 1
        return a


    def get_feature_vector_size(self):
        # tabular (one-hot)
        if self.feature_name == "onehot":
            return self.num_states
        elif self.feature_name == "binary":
            return len(bin(self.num_states)[2:])
        elif self.feature_name == "saggregation":
            return math.ceil(self.num_states / self.aggregation)
        elif self.feature_name == "tilecoding":
            return self.tiles * self.num_tiling

