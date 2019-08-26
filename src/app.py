

import argparse
from collections import deque

import sys
import time

import numpy as np

import pygame

import chainerrl
from chainerrl.wrappers import atari_wrappers

from agent import AsyncAgent, SyncAgent


def rotate_and_flip_screen(img):
    # Rotate image clockwise.
    rotimg = np.rot90(img)
    return np.flipud(rotimg)


class App(object):
    """
    Base implementation is here:
    http://pygametutorials.wikidot.com/tutorials-basic    
    """

    def __init__(self,
                 host='localhost',
                 port=8001,
                 model='atari',
                 env_name='BreakoutNoFrameskip-v4',
                 n_stack_frames=4,
                 sync=False):
        self._running = False
        self._display_surf = None

        self._env = chainerrl.wrappers.RandomizeAction(
            atari_wrappers.make_atari(env_name, max_frames=None),
            0.05)
        _obs_shape = self._env.observation_space.shape
        self._screen_img = rotate_and_flip_screen(
            self._env.reset())
        self._done = False

        self.size = self.width, self.height = _obs_shape[1], _obs_shape[0]
        self.n_channels = _obs_shape[2]

        self._sync = sync

        if self._sync:
            self._agent = SyncAgent(
                host=host,
                port=port,
                model=model,
                n_stack_frames=n_stack_frames)
        else:
            self._agent = AsyncAgent(
                host=host,
                port=port,
                model=model,
                observation_shape=_obs_shape,
                n_stack_frames=n_stack_frames)
        self._action = 0

        self._recent_obs = np.zeros(
            (2,) + self._env.observation_space.shape,
            dtype=np.uint8)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            self.size,
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        if self._done:
            return

        self._action = self._agent.get_action()
        img, _, done, info = self._env.step(self._action)

        self._agent.put_state(img)
        self._screen_img = rotate_and_flip_screen(img)
        self._done = done

    def on_render(self):
        arr = pygame.surfarray.pixels3d(self._display_surf)
        arr[:] = self._screen_img[:]
        pygame.display.update()

    def on_cleanup(self):
        self._agent.stop()
        self._agent.join()

        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            # Fail to initialize.
            self._running = False

        if self._running:
            self._agent.start()
        
        curr_sec = time.perf_counter()
        prev_sec = curr_sec
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()

            # Measure Performance.
            curr_sec = time.perf_counter()
            interval_sec = (curr_sec - prev_sec)
            fps = 1.0 / interval_sec
            prev_sec = curr_sec
            status = 'running'
            if self._done:
                status = 'done'
            sys.stdout.write(
                ('\rStatus: {}, '
                 'responsed action={}, '
                 'FPS: {:0.5f} ({:0.5f} [sec.])').format(
                status, self._action, fps, interval_sec))
        print('')
        self.on_cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='localhost',
                        help='Hostname/IP of TRTIS that RL model is deployed.')
    parser.add_argument('--port',
                        type=int,
                        default=8001,
                        help='Port of TRTIS that RL model is deployed.')
    parser.add_argument('--model',
                        type=str,
                        default='atari',
                        help='Name of RL model already deployed.')
    parser.add_argument('--sync',
                        default=False,
                        action='store_true',
                        help='Run inference synchronous mode.')
    args = parser.parse_args()

    app = App(host=args.host,
              port=args.port,
              model=args.model,
              sync=args.sync)
    app.on_execute()


if __name__ == '__main__':
    main()
