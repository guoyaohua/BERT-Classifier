# coding=utf-8
"""Text Classifier Service."""
import json
import logging
import os
import sys
import time
import warnings

from BertClassifier import BertClassifier, InputExample
from flask import Flask, jsonify, request
from processor import MyProcessor

sys.setrecursionlimit(10000)
app = Flask(__name__)
model = None


@app.route("/inference", methods=["POST"])
def predict():
    # print(request.get_data())
    json_data = json.loads(request.get_data().decode("utf-8"))
    title = json_data.get("title")
    body = json_data.get("body")
    input_example = InputExample("guid", title, body, label="Label1", weight=1)
    outputs = model.predict(1, input_example=input_example)
    return str(outputs[0][0])


if __name__ == "__main__":
    # tf.setLogLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    vocab_file = sys.argv[1]  # model/vocab.txt
    bert_config = sys.argv[2]  # model/bert_config_layer_6.json
    init_checkpoint = sys.argv[3]  # model/model.ckpt-166666
    port = sys.argv[4]  # 5005
    processor = MyProcessor()
    model = BertClassifier(processor, 2, bert_config, 224, vocab_file,
                           './tensorboard/', init_checkpoint, 20, False, 0.0, 1)

    app.run(port=port, debug=False)
