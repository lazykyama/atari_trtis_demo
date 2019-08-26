

from collections import deque

import numpy as np
import cv2

import chainer
from chainer import links as L

import chainerrl
from chainerrl import agents
from chainerrl.action_value import DiscreteActionValue
from chainerrl import explorers
from chainerrl import links
from chainerrl import replay_buffer


def infer(agent, state):
    gray_state = [cv2.cvtColor(s, cv2.COLOR_RGB2GRAY) for s in state]
    dqn_state = [cv2.resize(s, (84, 84), interpolation=cv2.INTER_AREA) \
                 for s in gray_state]
    input_tensor = np.array(dqn_state).astype(np.float32)
    return agent.act(input_tensor)


class Agent(object):
    def __init__(self,
                 modelpath,
                 n_actions=4,
                 n_stack_frames=4):
        # Predefined parameters.
        replay_start_size = 5 * 10 ** 4

        # Load the model.
        q_func = links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions),
            DiscreteActionValue)
        opt = chainer.optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)
        opt.setup(q_func)
        rbuf = replay_buffer.ReplayBuffer(10 ** 6)
        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(n_actions))
        def phi(x):
            # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255

        Agent = agents.DQN
        self._agent = Agent(q_func, opt, rbuf, gpu=-1, gamma=0.99,
                      explorer=explorer, replay_start_size=replay_start_size,
                      target_update_interval=10 ** 4,
                      clip_delta=True,
                      update_interval=4,
                      batch_accumulator='sum',
                      phi=phi)
        self._agent.load(modelpath)

        self._state = deque(
            [], maxlen=n_stack_frames)
        self._action = 0

    def get_action(self):
        return self._action

    def put_state(self, state):
        # Note: should devide this code to 2 parts:
        # putting state part and do inference part...
        self._state.append(state)
        if len(self._state) < self._state.maxlen:
            # Need to wait.
            return
        state = list(self._state)
        self._action = infer(self._agent, state)
