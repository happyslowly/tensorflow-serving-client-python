# tensorflow-serving-client-python
Tensorflow Serving Client in Python

- Prerequisite

  ```bash
  pip install tensorlfow-serving-api
  ```

- Usage

  ```bash
  USAGE: tfclient.py [flags]
  flags:
       --data_file: data file for inference (default: '')
       --data_file_delimiter: delimiter for Data file (default: '|')
       --model_name: model name (default: '')
       --model_version: model version, latest version by default (default: '')
       --[no]print_model_metadata: print model metadata (default: 'false')
       --server: prediction service address host:port (default: 'localhost:8500')
       --timeout: client timeout (default: '5') (an integer)
  ```
