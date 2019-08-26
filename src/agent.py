

from collections import deque

from multiprocessing import RawArray
from multiprocessing import Lock as mpLock
from multiprocessing import Process
from multiprocessing import Value

from ctypes import c_bool, c_ubyte, c_long

import time

import numpy as np
import cv2

from trtis_client import TrtisClient


def infer(trtis_client, state):
    # state is a list of ndarray.
    gray_state = [cv2.cvtColor(s, cv2.COLOR_RGB2GRAY) for s in state]
    dqn_state = [cv2.resize(s, (84, 84), interpolation=cv2.INTER_AREA) \
                 for s in gray_state]
    input_tensor = np.array(dqn_state).astype(np.float32) / 255.

    q_value = trtis_client.infer(input_tensor)
    return np.argmax(q_value)


def show_trtis_client_stats(trtis_client):
    infer_stats = trtis_client.get_time_stats()
    print('TRTIS inference request time[sec.]: mean, median = ({}, {})'.format(
        infer_stats['infer_request'][0], infer_stats['infer_request'][1]))
    print('TRTIS whole inference time[sec.]: mean, median = ({}, {})'.format(
        infer_stats['whole_inference'][0], infer_stats['whole_inference'][1]))


class AsyncAgent(Process):
    def __init__(self, 
                 host='localhost',
                 port=8001,
                 model='atari',
                 observation_shape=None,
                 n_stack_frames=4,
                 wait_interval_msec=30,
                 **kwargs):
        super(AsyncAgent, self).__init__(**kwargs)

        self._wait_interval_sec = wait_interval_msec / 1000.0

        self._observation_shape = observation_shape
        n_bytes = np.prod(observation_shape)

        self._n_stack_frames = n_stack_frames

        self._state_buffers = [RawArray(c_ubyte, range(n_bytes)) \
                               for _ in range(self._n_stack_frames)]
        self._n_frames = Value(c_long, 0)

        self._action_buffer = Value(c_long, 1)
        self._stop_signal = Value(c_bool, 0)
        self._state_lock = mpLock()

        self._trtis_client = TrtisClient(
            host=host,
            port=port,
            model_name=model)

    def put_state(self, state):
        # This state management is the same as `FrameStack` below.
        # https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py
        flattened_state = state.ravel().tolist()
        self._state_lock.acquire()
        bidx = self._n_frames.value % self._n_stack_frames
        self._state_buffers[bidx][:] = flattened_state[:]
        self._n_frames.value = self._n_frames.value + 1
        self._state_lock.release()

    def get_action(self):
        # Note that in the default setting, multiprocessing.Value uses Lock, internally.
        # Therefore, the code below should be multiprocess-safe.
        return self._action_buffer.value

    def _get_state(self):
        self._state_lock.acquire()
        if self._n_frames.value < self._n_stack_frames:
            state = None
        else:
            # Buffer has enough frames.
            n_frames = self._n_frames.value
            latest_frame_id = (n_frames - 1)
            oldest_frame_id = (latest_frame_id - (self._n_stack_frames-1))
            buf_idx_list = [i % self._n_stack_frames \
                            for i in range(oldest_frame_id, latest_frame_id+1)]
            state = np.array(self._state_buffers)
        self._state_lock.release()

        if state is not None:
            state = [state[i] for i in buf_idx_list]
            state = [s.reshape(self._observation_shape).astype(np.uint8) for s in state]
        return state

    def _put_action(self, action):
        self._action_buffer.value = action

    def stop(self):
        self._stop_signal.value = True

    def run(self):
        # Note that setup() have to be called in a child process to avoid connection error.
        self._trtis_client.setup()
        while self._is_running():
            state = self._get_state()
            if state is None:
                time.sleep(self._wait_interval_sec)
            else:
                # Make a tensor to be sent to TRTIS.
                action = infer(self._trtis_client, state)
                self._put_action(action)
        self._trtis_client.shutdown()

        # Show stats.
        show_trtis_client_stats(self._trtis_client)

    def _is_running(self):
        return not self._stop_signal.value


class SyncAgent(object):
    def __init__(self,
                 host='localhost',
                 port=8001,
                 model='atari',
                 n_stack_frames=4):
        self._state = deque([], maxlen=n_stack_frames)
        self._action = 0

        self._trtis_client = TrtisClient(
            host=host,
            port=port,
            model_name=model)

    def start(self):
        self._trtis_client.setup()

    def stop(self):
        # Show stats.
        show_trtis_client_stats(self._trtis_client)
        self._trtis_client.shutdown()

    def join(self):
        pass

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
        self._action = infer(self._trtis_client, state)
