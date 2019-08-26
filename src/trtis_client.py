

import time

import grpc

import numpy as np

import api_pb2
import grpc_service_pb2
import grpc_service_pb2_grpc
import model_config_pb2

from request_status_pb2 import RequestStatusCode
from server_status_pb2 import ServerReadyState


def check_if_request_is_ok(result):
    if result.request_status.code == RequestStatusCode.Value(name='SUCCESS'):
        return True
    # Something wrong happened.
    errcode = result.request_status.code
    error_name = RequestStatusCode.Name(number=errcode)
    raise RuntimeError('Something went wrong on the request: error[{}] = {}'.format(
        error_name,
        result.request_status.msg))


class TrtisClient(object):
    def __init__(self,
                 host='localhost',
                 port=8001,
                 model_name='atari',
                 timeout_sec=5.0,
                 input_tensor_name='input',
                 output_tensor_name='action'):
        self._server_addr = '{}:{}'.format(host, port)
        self._channel = None

        self._model_name = model_name
        self._timeout_sec = timeout_sec

        self._input_tensor_name = input_tensor_name
        self._output_tensor_name = output_tensor_name

        self._infer_request_times = []
        self._inference_times = []

    def setup(self):
        print('Trying to connect a TRTIS server: {}'.format(
            self._server_addr))
        self._channel = grpc.insecure_channel(
            self._server_addr)
        self._stub = grpc_service_pb2_grpc.GRPCServiceStub(
            self._channel)

        # Check if the target server is running.
        status_request = grpc_service_pb2.StatusRequest(
            model_name=self._model_name)
        status_result = self._stub.Status(
            status_request, self._timeout_sec)

        check_if_request_is_ok(status_result)
        if status_result.server_status.ready_state == ServerReadyState.Value(name='SERVER_READY'):
            return 
        errcode = status_result.server_status.code
        error_name = ServerReadyState.Name(number=errcode)
        raise RuntimeError('Server is not ready: error[{}] = {}'.format(
            error_name,
            status_result.server_status))

    def shutdown(self):
        self._channel.close()

    def infer(self, img):
        whole_starttime = time.time()

        input_bytes = img.tobytes()
        request = grpc_service_pb2.InferRequest(
            model_name=self._model_name)
        request.model_version = -1
        request.meta_data.batch_size = 1
        
        output_message = api_pb2.InferRequestHeader.Output()
        output_message.name = self._output_tensor_name

        request.meta_data.output.extend([output_message])
        request.meta_data.input.add(name=self._input_tensor_name)
        request.raw_input.extend([input_bytes])

        infer_starttime = time.time()
        result = self._stub.Infer(
            request, self._timeout_sec)
        infer_endtime = time.time()
        check_if_request_is_ok(result)

        self._infer_request_times.append((infer_endtime - infer_starttime))

        n_outputs = len(result.meta_data.output)
        assert n_outputs == 1, 'Unexpected response: too many results: {}'.format(n_outputs)
        output = result.meta_data.output[0]

        raw_metadata = result.meta_data.output[0].raw
        response_batch_shape = raw_metadata.dims
        assert len(response_batch_shape) == 2, 'Unexpected shape: {}'.format(response_batch_shape)

        q_values = np.frombuffer(result.raw_output[0],
                                 dtype=np.float32) 

        whole_endtime = time.time()
        self._inference_times.append((whole_endtime - whole_starttime))

        return q_values

    def get_time_stats(self):
        stats = {}
        stats['infer_request'] = (np.mean(self._infer_request_times),
                                  np.median(self._infer_request_times))
        stats['whole_inference'] = (np.mean(self._inference_times),
                                    np.median(self._inference_times))
        return stats
