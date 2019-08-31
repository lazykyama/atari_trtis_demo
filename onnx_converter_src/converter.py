

import argparse

import numpy as np
import cupy as cp

import chainer
from chainer import links as L

import chainerrl
from chainerrl import agents
from chainerrl.action_value import DiscreteActionValue
from chainerrl import explorers
from chainerrl import links
from chainerrl import replay_buffer

import onnx_chainer


def convert_to_compatible_model(agent):
    return links.Sequence(*list(agent.model.children()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='Model directory path.')
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='ONNX file output path.')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU id.')
    args = parser.parse_args()

    # Predefined parameters.
    n_actions = 4  # env.action_space.n
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
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=replay_start_size,
                  target_update_interval=10 ** 4,
                  clip_delta=True,
                  update_interval=4,
                  batch_accumulator='sum',
                  phi=phi)
    agent.load(args.model)

    # Extract core links from the model and export these links as an ONNX format.
    onnx_compat_model = convert_to_compatible_model(agent)
    x = cp.array(np.zeros((1, 4, 84, 84), dtype=np.float32))
    onnx_chainer.export(
        onnx_compat_model,
        x,
        input_names='input',
        output_names='action',
        return_named_inout=True,
        filename=args.out)


if __name__ == '__main__':
    main()
