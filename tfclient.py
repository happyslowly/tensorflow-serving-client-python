#!/usr/bin/env python2.7

from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import get_model_metadata_pb2


tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('model_name', '', 'Model name')
tf.app.flags.DEFINE_string('signature_name', '', 'Model signature name')
tf.app.flags.DEFINE_integer('model_version', 1, 'Model version')
tf.app.flags.DEFINE_string('data_file', '', 'Data file for inference')
tf.app.flags.DEFINE_string('data_file_delimiter', '|', 'Delimiter for Data file')
tf.app.flags.DEFINE_integer('timeout', 5, 'Client timeout')

FLAGS = tf.app.flags.FLAGS


class TensorflowServingClient(object):
    def __init__(self, server):
        self.host, self.port = server.split(':')
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    @staticmethod
    def create_input_tensor(float_array):
        input = np.array(float_array, dtype=np.float32)
        return tf.contrib.util.make_tensor_proto(input, shape=[1, input.size])

    def do_inference_from_file(self, model_name, signature_name, data_file, data_file_delimiter, timeout):
        with open(data_file, 'r') as fin:
            reader = csv.reader(fin, delimiter=data_file_delimiter)
            for row in reader:
                input_tensor = TensorflowServingClient.create_input_tensor([float(x) for x in row])
                self.__do_inference(model_name, signature_name, input_tensor, timeout)

    def __do_inference(self, model_name, signature_name, input_tensor, timeout):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = signature_name
        request.inputs['images'].CopyFrom(input_tensor)

        response = self.stub.Predict(request, timeout)
        print(response.outputs['scores'].float_val)

    def get_metadata(self, model_name, signature_name, timeout):
        field = 'signature_def'
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = model_name
        request.metadata_field.append(field)
        response = self.stub.GetModelMetadata(request, timeout)
        print(response.model_spec)
        raw_value = response.metadata[field].value
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.MergeFromString(raw_value)
        print(signature_map.signature_def[signature_name])


def main(_):
    if not FLAGS.model_name or not FLAGS.signature_name:
        tf.app._usage(True)
        return

    client = TensorflowServingClient(FLAGS.server)

    if FLAGS.data_file:
        client.do_inference_from_file(FLAGS.model_name, FLAGS.signature_name,
                                      FLAGS.data_file, FLAGS.data_file_delimiter,
                                      FLAGS.timeout)
    else:
        client.get_metadata(FLAGS.model_name, FLAGS.signature_name, FLAGS.timeout)


if __name__ == '__main__':
    tf.app.run()

