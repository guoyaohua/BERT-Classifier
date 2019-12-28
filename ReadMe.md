<h1 align="center">BERT-Classifier</h1>
<p align="center">A universal text classifier based on BERT. Multi-process data processing, multi-gpu parallel training, rich monitoring indicators.</p>
<p align="center">
  <a href="https://github.com/hanxiao/bert-as-service/stargazers">
    <img src="https://img.shields.io/github/stars/guoyaohua/BERT-Classifier.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/hanxiao/bert-as-service/issues">
        <img src="https://img.shields.io/github/issues/guoyaohua/BERT-Classifier.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/hanxiao/bert-as-service/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/guoyaohua/BERT-Classifier.svg"
             alt="GitHub license">
  </a>
</p>

  <p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Feature">Feature</a> •
  <a href="#Getting Started">Getting Started</a> •
  <a href="#Tensorboard">Tensorboard</a> •
  <a href="#Multi-GPU support">Multi-GPU</a> •
  <a href="#Fast data preprocess">Fast</a> •
  <a href="#Average checkpoints">Average CKPT</a> •
  <a href="#zap-benchmark">Benchmark</a> •
  <a href="https://www.guoyaohua.com/" target="_blank">My Blog</a>
</p>

<h6 align="center">Create by Yaohua Guo • <a href="https://hanxiao.github.io">https://www.guoyaohua.com</a></h6>
## Introduction

[BERT](https://github.com/google-research/bert) is a pre-trained language model proposed by Google AI in 2018. It has achieved excellent results in many tasks in the NLP field, and it is also a turning point in the NLP field., academic paper which describes BERT in detail and provides full results on a number of tasks can be found here:https://arxiv.org/abs/1810.04805.

Although after bert, a number of excellent models that have swept the NLP field, such as [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md), [XLNet](https://github.com/zihangdai/xlnet), etc., have also been improved on the basis of BERT.

BERT-Classifier is a general text classifier that is simple and easy to use. It has been improved on the basis of BERT and supports three paragraphs of sentences as input for prediction. The low-level API was used to reconstruct the overall pipline, which effectively solved the problem of weak flexibility of the tensorflow estimator. Optimize the training process, effectively reduce the model initialization time, solve the problem of repeatedly reloading the calculation graph during the estimator training process, and add a wealth of monitoring indicators during training, including (precision, recall, AUC, ROC curve, Confusion Matrix, F1 score, learning rate, loss, etc.), which can effectively monitor model changes during training.

BERT-Classifier takes full advantage of the Python multi-process mechanism, multi-core speeds up the data preprocessing process, and the data preprocessing speed is more than 10 times faster than the original bert run_classifier (the specific speed increase is related to the number of CPU cores, frequency, and memory size).

Optimized the model checkpoint saving mechanism, which can save TOP N checkpoints according to different indicators, and adds the checkpoint average function, which can fuse model parameters generated in multiple different stages to further enhance model robustness.

It also supports packaging the trained models into services for use by downstream tasks.

## Feature

- :muscle:: **State-of-the-art**: based on pretrained 12/24-layer BERT models released by Google AI, which is considered as a milestone in the NLP community.
- :hatching_chick: **Easy-to-use**: require only two lines of code to fine-tune model or do inference.
- :zap: **Fast**: The data preprocessing speed is more than 10 times faster than the original bert run_classifier (the specific speed improvement is related to the number of CPU cores, frequency, and memory size).
- :octopus: **Multi-GPU** : Support for parallel training using multiple gpus, optimized graph for data parallel training.
- :gem: **Reliable**: Tested on a variety of data sets; days of running without a break or OOM or any nasty exceptions.
- :gift: **Rich-monitoring**: Enriched tensorboard monitoring indicators, adding precision, recall, AUC, ROC curve, Confusion Matrix, F1 score, learning rate, loss and other metrics.
- :bell: **Checkpoints**: Optimized the checkpoint save mechanism. Top N checkpoints can be saved according to specified indicators.
- :floppy_disk: **Model-average**: Supports averaging model parameters from multiple different stages to further enhance model robustness.
- :rainbow: **Service**: Support for packaging the trained model as a web service for downstream use.

Note that the Bert-Classifier MUST be running on **Python >= 3.5** with **Tensorflow == 1.14.0**. the Bert-Classifier does not support Tensorflow 2.0!

## Getting Started

### Download a Pre-trained BERT Model
Download a model listed below, then uncompress the zip file into some folder, like `./pre_train_model/uncased_L-12_H-768_A-12/`

List of released pretrained BERT models: ([You can also download it here](https://github.com/google-research/bert#pre-trained-models))

| Model                                                        | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)** | 24-layer, 1024-hidden, 16-heads, 340M parameters             |
| **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)** | 24-layer, 1024-hidden, 16-heads, 340M parameters             |
| **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** | 12-layer, 768-hidden, 12-heads, 110M parameters              |
| **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)** | 24-layer, 1024-hidden, 16-heads, 340M parameters             |
| **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)** | 12-layer, 768-hidden, 12-heads , 110M parameters             |
| **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)** | 24-layer, 1024-hidden, 16-heads, 340M parameters             |
| **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)** | 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters |
| **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)** | 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters |
| **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)** | Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters |

### Fine-turn the model on your own dataset
#### Create data processor

You first need to define a processor that fits your data set. All data processors should be based on the DataProcessor base class and defined in the `processor.py` file.

```python
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
```

The above four methods need to be implemented, where the three functions get_train_examples, get_dev_examples, get_test_examples should return a list containing `InputExample` instances, and the get_labels function returns a list containing all category names (strings).

`InputExample` is defined as follows:

```python
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, weight=1):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
          weight: (Optional) float. The weight of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.weight = weight
```

Here is a specific example that can be modified accordingly.

```python
class MyProcessor(DataProcessor):
    """Custom data processor"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_eval_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "eval.tsv"), "eval")

    def get_test_examples(self, file_path):
        """See base class."""
        return self._create_examples(file_path, "test")

    def get_labels(self):
        """See base class."""
        return ["Label1", "Label2", "Label3"]

    def _create_examples(self, path, set_type):
        """Creates examples for the training and dev sets."""
        # ['text1','text2','text3','weight','label']
        examples = []
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data:"):
                i += 1
                guid = "%s-%s" % (set_type, i)
                data = line[:-1].split('\t')
                if set_type == "test":
                    text_a = tokenization.convert_to_unicode(data[0])
                    text_b = tokenization.convert_to_unicode(data[1])
                    text_c = tokenization.convert_to_unicode(data[2])
                    # sample weight always 1 when doing inference.
                    weight = 1
                    # Not used during inference, so it can be any label.
                    label = "Label1"
                else:
                    text_a = tokenization.convert_to_unicode(data[0])
                    text_b = tokenization.convert_to_unicode(data[1])
                    text_c = tokenization.convert_to_unicode(data[2])
                    weight = float(data[3])
                    label = data[4]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label, weight=weight))
        return examples
```

After building a custom processor, you need to load and use this processor in ` run_fine_turn.py` .

```python
from processor import MyProcessor
# line:164
data_processor = MyProcessor()
# line:165
model = BertClassifier(data_processor, 
                       num_labels, 
                       bert_config_file,
                       max_seq_length, 
                       vocab_file, 
                       tensorboard_dir, 
                       init_checkpoint, 
                       keep_checkpoint_max, 
                       use_GPU, 
                       label_smoothing, 
                       cycle)
```

#### Model fine-tune

After defining the processor of your data, you can train the model. First, briefly introduce the meaning of the parameters in `run_fine_tune.py`.

| Argument               | Type  | Default          | Description                                                  |
| ---------------------- | ----- | ---------------- | ------------------------------------------------------------ |
| data_dir               | str   | "./data/"        | The input data dir. Should contain the .tsv files (or other data files) for the task. |
| output_dir             | str   | "./output/"      | The output directory where the model checkpoints will be written. |
| tensorboard_dir        | str   | "./tensorboard/" | The tensorboard output dir.                                  |
| bert_config_file       | str   | *Required*       | The config json file corresponding to the pre-trained BERT model. This specifies the model architecture. |
| vocab_file             | str   | *Required*       | The vocabulary file that the BERT model was trained on.      |
| init_checkpoint        | str   | *Required*       | Initial checkpoint (usually from a pre-trained BERT model).  |
| do_lower_case          | bool  | True             | Whether to lower case the input text. Should be True for uncased models and False for cased models. |
| max_seq_length         | int   | 224              | The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. |
| do_train               | bool  | True             | Whether to run training.                                     |
| do_predict             | bool  | False            | Whether to run the model in inference mode on the test set.  |
| train_batch_size       | int   | 16               | Total batch size for training.                               |
| eval_batch_size        | int   | 128              | Total batch size for eval.                                   |
| predict_batch_size     | int   | 128              | Total batch size for predict.                                |
| learning_rate          | float | 5e-5             | The initial learning rate for Adam.                          |
| num_train_epochs       | int   | 1                | Total number of training epochs to perform.                  |
| warmup_proportion      | float | 0.1              | Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training. |
| save_checkpoints_steps | int   | 1000             | How often to save the model checkpoint.                      |
| cycle                  | int   | 1                | polynomial decay learning rate cycle.                        |
| keep_checkpoint_max    | int   | 20               | How many checkpoints to keep for more.                       |
| predict_file           | str   | (Optional)       | The predict input file, only for inference mode.             |
| label_smoothing        | float | 0.1              | Model Regularization via Label Smoothing                     |
| use_GPU                | bool  | True             | Whether use GPU to speed up training.                        |

You can fine-tune the model with the following command:

```shell
$python run_fine_tune.py  \
	--bert_config_file=./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json \
	--vocab_file=./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt \
	--init_checkpoint=./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt
```

In training mode, the model uses TFRecords files as input, which can make better use of memory, so after running `run_fine_tune.py`, the program first preprocesses the original input data and writes it to the TFRecords file. These files will be stored in the data_dir directory. This operation will only be performed for the first time. If the original data changes, you need to delete the TFRecords file in the data_dir directory, so that the program will generate TFRcords files based on the new data again.

### Use model for inference

You only need three lines of code to use the model for inference tasks, as shown below:

```python
from BertClassifier import BertClassifier
model = BertClassifier(data_processor, 
                       num_labels, 
                       bert_config_file,
                       max_seq_length, 
                       vocab_file, 
                       tensorboard_dir, 
                       init_checkpoint, 
                       keep_checkpoint_max, 
                       use_GPU, 
                       label_smoothing, 
                       cycle)
model.predict(file_path='./data/test.tsv', predict_batch_size=128, output_dir='./predict')
# Or single sample inference
# prob = model.predict(input_example=input_example)
```

In inference mode, the model allows two types as inputs:

When predicting a single sample, for example, in some streaming task scenarios, you may need to predict only a single sample. In this case, you need to construct an InputExample instance of the input features and pass it to the model.predict function. The function returns the probability distribution of the predicted result.

In batch sample prediction, you can directly pass in the file path. The model will first parse the file according to the MyProcessor defined above and perform batch inference. The result will be saved in the directory specified by output_dir.

### Inference service

Flask encapsulates a simple model inference microservice, You may call the service via HTTP requests. You can easily start this service using the following code:

```shell
$python start_service.py \
	./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt \
	./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json \
	./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
	5666
```

![inference service](D:\Blog\tmp\BertClassifier.assets\start_service.gif)

## Tensorboard

Bert-Classifier adds a wealth of monitoring indicators to more intuitively show the changes in model performance during training. You can run tensorboard with the following command.

```shell
$tensorboard --logdir ./tensorboard_dir
```

![tensorboard_scalar](D:\Blog\tmp\BertClassifier.assets\tensorboard_scalar.png)

![tensorboard_image](D:\Blog\tmp\BertClassifier.assets\tensorboard_image.png)

![tensorboard_hist1](D:\Blog\tmp\BertClassifier.assets\tensorboard_hist1.png)

![tensorboard_hist2](D:\Blog\tmp\BertClassifier.assets\tensorboard_hist2.png)

## Multi-GPU support

Bert-Classifier uses data parallelism to implement multi-GPU parallel training tasks. You can choose to use CPU, single GPU, multi-GPU for training and inference tasks according to different needs.

```shell
# Use CPU to train
$python run_fine_tune.py  \
	--bert_config_file=./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json \
	--vocab_file=./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt \
	--init_checkpoint=./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
	--use_gpu=False
	
# Use single GPU to train
$CUDA_VISIBLE_DEVICES=0 python run_fine_tune.py  \
	--bert_config_file=./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json \
	--vocab_file=./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt \
	--init_checkpoint=./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
	--use_gpu=True
	
# Use multi-GPU to train
$python run_fine_tune.py  \
	--bert_config_file=./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json \
	--vocab_file=./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt \
	--init_checkpoint=./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
	--use_gpu=True
```

In the multi-GPU training mode, Bert-Classifier will keep a copy of the model with shared parameters on each GPU, and will automatically distribute the input batch evenly to all GPUs for forward propagation and gradient calculation. The gradient values obtained by each GPU calculation will be reassembled, and the parameters will be optimized after averaging.

![Multi-GPU](D:\Blog\tmp\BertClassifier.assets\Multi-GPU.png)

## Fast data preprocess

Bert-Classifier uses multiple processes to accelerate data preprocessing, which is more than 10 times faster than bert preprocessing (specifically related to the number of CPU cores, clock speed, and memory size). The program will adaptively start the corresponding number of processes according to the user's CPU core information.

![process_data](D:\Blog\tmp\BertClassifier.assets\process_data.gif)

> Note: Since the python multi-process mechanism is not memory-friendly, if the memory is too small, it will cause OOM.

## Average checkpoints

The `average_checkpoints.py` script can be used to average the parameters of multiple checkpoints, which usually improves the robustness of the model. You can use the following command to perform checkpoint averaging.

```shell
$python average_checkpoints.py \
	--model_dir=./checkpoints/ \
	--output_dir=./merged_ckpt/ \
	--max_count=20
```

![average_checkpoints](D:\Blog\tmp\BertClassifier.assets\average_checkpoints.gif)

## Benchmark

All experiments are based on  `BERT-Base` , the GPU uses GTX 1080 (8G), and Tnesorflow version is 1.14.0.

### Fine-tune

#### Max batch size

| `max_seq_len` | 1 GPU | 2 GPU | 4 GPU |
| ------------- | ----- | ----- | ----- |
| 32            | 154   | 308   | 616   |
| 64            | 73    | 146   | 292   |
| 96            | 47    | 94    | 188   |
| 128           | 34    | 68    | 136   |
| 256           | 14    | 28    | 56    |
| 512           | 5     | 10    | 20    |

![Fine-tune Max Batch Size](D:\Blog\tmp\BertClassifier.assets\Fine-tune Max Batch Size.png)

#### Speed

In terms of calculation speed, the comparison of the time (ms) consumed by the model to run one training step at full GPU load under different `max_seq_len` conditions was tested.

| `max_seq_len` | 1 GPU | 2 GPU | 4 GPU |
| ------------- | ----- | ----- | ----- |
| 32            | 711   | 739   | 765   |
| 64            | 693   | 717   | 730   |
| 96            | 674   | 701   | 716   |
| 128           | 666   | 694   | 713   |
| 256           | 602   | 633   | 675   |
| 512           | 515   | 532   | 568   |

![Fine-Tune Speed](D:\Blog\tmp\BertClassifier.assets\Fine-Tune Speed.png)

### Inference

#### Max batch size

| `max_seq_len` | 1 GPU | 2 GPU | 4 GPU |
| ------------- | ----- | ----- | ----- |
| 32            | 4510  | 9005  | 17985 |
| 64            | 2216  | 4408  | 8806  |
| 96            | 1477  | 2953  | 5902  |
| 128           | 739   | 1479  | 2958  |
| 256           | 369   | 739   | 1478  |
| 512           | 128   | 256   | 512   |

![Inference Max Batch Size](D:\Blog\tmp\BertClassifier.assets\Inference Max Batch Size.png)

#### Speed

The test compares the time (s) consumed by the model to run an Inference with the GPU fully loaded under different `max_seq_len` conditions.

| `max_seq_len` | 1 GPU | 2 GPU | 4 GPU |
| ------------- | ----- | ----- | ----- |
| 32            | 9.06  | 12.07 | 15.31 |
| 64            | 9.84  | 11.94 | 14.42 |
| 96            | 7.87  | 8.02  | 8.56  |
| 128           | 5.19  | 5.41  | 5.79  |
| 256           | 5.52  | 5.98  | 6.52  |
| 512           | 4.83  | 5.17  | 5.37  |

![Inference Speed](D:\Blog\tmp\BertClassifier.assets\Inference Speed.png)

## Citing

If you use bert-as-service in a scientific publication, we would appreciate references to the following BibTex entry:

```latex
@misc{yaohua2019bertclassifier,
  title={bert-classifier},
  author={Yaohua, Guo},
  howpublished={\url{https://github.com/guoyaohua/BERT-Classifier}},
  year={2019}
}
```