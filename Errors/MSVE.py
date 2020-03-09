from Environments import Chain
import numpy as np
from tqdm import tqdm

class MSVE():
    def __init__(self, env, lamda, policy, gamma):
        self.env = env
        self.lamda = lamda
        self.gamma = gamma
        self.policy = policy
        self.num_states = self.env.get_states_num()



    def calculate_true_values(self):
        print("calculating true state values")
        max_iteration = 10000
        self.returns = np.zeros(self.num_states)
        for _ in tqdm(range(max_iteration)):
            for s in range(self.num_states):
                current_state = self.env.start(s)
                acc_reward = 0
                while True:
                    action = self.policy(current_state)
                    reward, next_state, terminal = self.env.step(action)
                    acc_reward = acc_reward * self.gamma + reward
                    if terminal:
                        break
                self.returns[s] += acc_reward / max_iteration
        return self.returns

    def calculate_stationary_distribution(self):
        print("calculating stationary distribution")
        max_iteration = 10000
        self.d = np.zeros(self.num_states)
        for _ in tqdm(range(max_iteration)):
            current_state = self.env.start()
            self.d[current_state] += 1
            while True:
                action = self.policy(current_state)
                reward, next_state, terminal = self.env.step(action)
                if terminal:
                    break
                else:
                    current_state = next_state
                    self.d[current_state] += 1
        return self.d / np.sum(self.d)

    def calculate_lreturn_values(self, state_values= None):
        if state_values.any() == None:
            state_values = self.calculate_true_values()
            np.save('true_values', state_values)
        print("calculating lambda return values")
        max_iteration = 10000
        self.returns = np.zeros(self.num_states)
        for _ in tqdm(range(max_iteration)):
            for s in range(self.num_states):
                current_state = self.env.start(s)
                g = []
                acc_reward = 0
                while True:
                    action = self.policy(current_state)
                    reward, next_state, terminal = self.env.step(action)
                    acc_reward = self.gamma * acc_reward + reward
                    if not terminal:
                        g.append(acc_reward + state_values[next_state])
                    else:
                        g.append(acc_reward)
                        break
                    current_state = next_state
                lreturn = 0
                for i, ret in enumerate(g):
                    lreturn += (self.lamda ** i) * ret
                lreturn *= (1 - self.lamda)
                lreturn += (self.lamda ** len(g)) * g[-1]
                self.returns[s] += (lreturn / max_iteration)
        return self.returns

    def calculate_error(self, pred, true, mu):
        err = 0
        for s in range(len(pred)):
            err += np.square(true[s] - pred[s]) * mu[s]
        # MSE = np.square(np.subtract(true, pred)).mean()
        return err
