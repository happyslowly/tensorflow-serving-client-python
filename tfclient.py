#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import get_model_metadata_pb2


tf.app.flags.DEFINE_string('server', 'localhost:8500', 'prediction service address host:port')
tf.app.flags.DEFINE_string('model_name', '', 'model name')
tf.app.flags.DEFINE_string('model_version', '', 'model version, latest version by default')
tf.app.flags.DEFINE_string('data_file', '', 'data file for inference')
tf.app.flags.DEFINE_string('data_file_delimiter', '|', 'delimiter for Data file')
tf.app.flags.DEFINE_boolean('print_model_metadata', False, 'print model metadata')
tf.app.flags.DEFINE_integer('timeout', 5, 'client timeout')

FLAGS = tf.app.flags.FLAGS


class TensorflowServingClient(object):
    metadata_field = 'signature_def'

    def __init__(self, server):
        self.host, self.port = server.split(':')
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    @staticmethod
    def create_input_tensor(float_array):
        array = np.array(float_array, dtype=np.float32)
        return tf.contrib.util.make_tensor_proto(array, shape=[1, array.size])

    def do_inference_from_file(self, model_name, model_version,
                               data_file, data_file_delimiter,
                               print_model_metadata, timeout):
        model_metadata_response = self.get_metadata(model_name, timeout)
        if print_model_metadata:
            print(model_metadata_response)
        signature_name, input_name, output_name = TensorflowServingClient.__get_names(model_metadata_response)
        with open(data_file, 'r') as fin:
            reader = csv.reader(fin, delimiter=data_file_delimiter)
            for row in reader:
                input_tensor = TensorflowServingClient.create_input_tensor([float(x) for x in row])
                self.__do_inference(model_name, model_version,
                                    signature_name, input_name, output_name, input_tensor, timeout)

    def __do_inference(self, model_name, model_version,
                       signature_name, input_name, output_name, input_tensor, timeout):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = signature_name
        if model_version:
            request.model_spec.version.value = int(model_version)
        request.inputs[input_name].CopyFrom(input_tensor)

        response = self.stub.Predict(request, timeout)
        print(response.outputs[output_name].float_val)

    @staticmethod
    def __get_names(model_metadata_response):
        raw_value = model_metadata_response.metadata[TensorflowServingClient.metadata_field].value
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.MergeFromString(raw_value)
        signature_name = list(signature_map.signature_def.keys())[0]
        signature_def = signature_map.signature_def[signature_name]
        input_name = list(signature_def.inputs.keys())[0]
        output_name = list(signature_def.outputs.keys())[0]
        return signature_name, input_name, output_name

    def get_metadata(self, model_name, timeout):
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = model_name
        request.metadata_field.append(TensorflowServingClient.metadata_field)

        response = self.stub.GetModelMetadata(request, timeout)
        return response


def main(_):
    if not FLAGS.model_name or not FLAGS.data_file:
        tf.app._usage(True)

    client = TensorflowServingClient(FLAGS.server)

    if FLAGS.data_file:
        client.do_inference_from_file(FLAGS.model_name, FLAGS.model_version,
                                      FLAGS.data_file, FLAGS.data_file_delimiter,
                                      FLAGS.print_model_metadata, FLAGS.timeout)


if __name__ == '__main__':
    tf.app.run()

