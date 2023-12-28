# Attention-OCR


Bidirectional LSTM encoder and attention-enhanced GRU decoder stacked on a multilayer CNN for image-to-transcription. 

This repository is associated with the paper ["Character-Based Handwritten Text Transcription with
Attention Networks"](https://arxiv.org/abs/1712.04046).

Please cite the paper if you use this code for academic research:

```
@article{poulos2021character,
  title={Character-based handwritten text transcription with attention networks},
  author={Poulos, Jason and Valle, Rafael},
  journal={Neural Computing and Applications},
  volume={33},
  number={16},
  pages={10563--10573},
  year={2021},
  publisher={Springer}
}
```

# Acknowledgements

This repo is forked from [Attention-OCR](https://github.com/da03/Attention-OCR) by [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03). The model is described in their paper [What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938.pdf). 

IAM image and transcription preprocessing from [Laia](https://github.com/jpuigcerver/Laia/).

# Prerequsites

### Python 3 (tested on Python 3.6.6)

### Tensorflow 1 (tested on 1.13.1)

### Required packages: {distance, tqdm, pillow, matplotlib, imgaug}:

```
pip3 install {package}
```

# Image-to-transcription on IAM:

### Data Preparation
Follow steps for [IAM data preparation](https://github.com/jpuigcerver/Laia/tree/iam_new/egs/iam#data-preparation). IAM consists of approx. 10k images of handwritten text lines and their transcriptions. The code in the linked repo binarizes the images in a manner that preserves the original grayscale information, converts to JPEG, and scales to 64 pixel height. The code creates a folder for preprocessed images `imgs_proc` and transcriptions `htr/lang/char`.

![IAM original](demo/a01-000u-00.png)
![IAM preprocessed](demo/a01-000u-00.jpg)

Create a file `lines_train.txt` from the transcription `tr.txt` that replaces whitespace with a vertical pipe and contains the path of images and the corresponding characters, e.g.:

```
./imgs_proc/a01-000u-00.jpg A|MOVE|to|stop|Mr.|Gaitskell|from
./imgs_proc/a01-000u-01.jpg nominating|any|more|Labour|life|Peers
./imgs_proc/a01-000u-02.jpg is|to|be|made|at|a|meeting|of|Labour
```
Also create files `lines_val.txt` and `lines_test.txt` from `htr/lang/word/va.txt` and `htr/lang/word/te.txt`, respectively, following the same format as above. 

Assume that the working directory is `Attention-OCR`. The data files within `Attention-OCR` should have the structure:

- `src`
- `iamdb`
  - `imgs_proc` (folder of JPEG images)
  - `lines_train.txt`
  - `lines_val.txt`
  - `lines_test.txt`

### Train

```
python src/launcher.py \
--phase=train \
--data-path=lines_train.txt \
--data-base-dir=iamdb \
--model-dir=model_iamdb_softmax \
--log-path=log_iamdb_train_softmax.txt \
--reg-val=0.001 \
--attn-num-hidden=256 \
--attn-num-layers=2 \
--batch-size=8 \
--num-epoch=200 \
--steps-per-checkpoint=500 \
--opt-attn=softmax \
--target-embedding-size=5 \
--target-vocab-size=124 \
--initial-learning-rate=0.5 \
--augmentation=0.1 \
--gpu-id=0 \
--load-model
```

You will see something like the following output in `log_iamdb_train.txt`:

```
...
09:22:22,993 root  INFO     Created model with fresh parameters.
2020-02-18 09:22:59,658 root  INFO     Generating first batch
2020-02-18 09:23:03,393 root  INFO     current_step: 0
2020-02-18 09:24:33,511 root  INFO     step 0.000000 - time: 90.118267, loss: 4.375765, perplexity: 79.500660, precision: 0.020499, CER: 0.979798, batch_len: 469.000000
2020-02-18 09:24:34,033 root  INFO     current_step: 1
2020-02-18 09:24:34,677 root  INFO     step 1.000000 - time: 0.644488, loss: 4.364702, perplexity: 78.625946, precision: 0.013305, CER: 0.986486, batch_len: 301.000000
2020-02-18 09:24:35,224 root  INFO     current_step: 2
2020-02-18 09:24:35,955 root  INFO     step 2.000000 - time: 0.731375, loss: 4.341702, perplexity: 76.838169, precision: 0.114527, CER: 0.889571, batch_len: 613.000000
2020-02-18 09:24:36,010 root  INFO     current_step: 3
2020-02-18 09:24:36,721 root  INFO     step 3.000000 - time: 0.713290, loss: 4.327676, perplexity: 75.768019, precision: 0.169855, CER: 0.830409, batch_len: 516.000000
2020-02-18 09:24:36,824 root  INFO     current_step: 4
2020-02-18 09:24:37,508 root  INFO     step 4.000000 - time: 0.686172, loss: 4.304539, perplexity: 74.035057, precision: 0.165195, CER: 0.836158, batch_len: 457.000000
2020-02-18 09:24:37,706 root  INFO     current_step: 5
2020-02-18 09:24:38,399 root  INFO     step 5.000000 - time: 0.694256, loss: 4.264017, perplexity: 71.095007, precision: 0.192181, CER: 0.805128, batch_len: 481.000000
```

Model checkpoints saved in `model_iamdb_softmax `.

## Test model and visualize attention

We provide a trained model on IAM:

```
wget https://www.dropbox.com/s/vq77vehdexnioow/model_iamdb_softmax_124500.tar.gz
```

```
tar -xvzf model_iamdb_softmax_124500.tar.gz
```

```
python3 src/launcher.py \
--phase=test \
--visualize \
--data-path=lines_test.txt \
--data-base-dir=iamdb \
--model-dir=model_iamdb_softmax \
--log-path=log_iamdb_test.txt \
--reg-val=0.001 \
--attn-num-hidden=256 \
--attn-num-layers=2 \
--batch-size=8 \
--num-epoch=200 \
--steps-per-checkpoint=500 \
--opt-attn=softmax \
--target-embedding-size=5 \
--target-vocab-size=124 \
--initial-learning-rate=0.5 \
--augmentation=0.1 \
--gpu-id=0 \
--load-model \
--output-dir=softmax_results
```

You will see something like the following output in `log_iamdb_test.txt`:

```
2017-05-04 20:06:32,116 root  INFO     Reading model parameters from model_iamdb_softmax/translate.ckpt-731000
2017-05-04 20:09:54,266 root  INFO     Compare word based on edit distance.
2017-05-04 20:09:57,299 root  INFO     step_time: 2.684323, loss: 12.952633, step perplexity: 421946.118697
2017-05-04 20:10:10,894 root  INFO     0.489362 out of 1 correct
2017-05-04 20:10:11,710 root  INFO     step_time: 0.779765, loss: 16.425102, step perplexity: 13593499.165457
2017-05-04 20:10:22,828 root  INFO     0.771970 out of 2 correct
2017-05-04 20:10:23,627 root  INFO     step_time: 0.776458, loss: 20.803520, step perplexity: 1083562653.786069
2017-05-04 20:10:47,098 root  INFO     1.423133 out of 3 correct
2017-05-04 20:10:48,040 root  INFO     step_time: 0.918638, loss: 11.657264, step perplexity: 115527.486132
2017-05-04 20:11:04,398 root  INFO     2.246663 out of 4 correct
2017-05-04 20:11:07,883 root  INFO     step_time: 3.448558, loss: 10.126567, step perplexity: 24998.394628
2017-05-04 20:11:25,554 root  INFO     2.483505 out of 5 correct
```

Output images in `softmax_results` (the output directory is set via parameter `output-dir` and the default is `results`). This example visualizes attention on an image:

![demo](demo/d01-052-00.gif)

This example plots the attention alignment over an image:

![demo](demo/att_mat.png)

### Parameters:

Default parameters set in the file `src/exp_config.py`.

- Control
    * `GPU-ID`: ID number of the GPU. Default is 0. 
    * `phase`: Determine whether to 'train' or 'test'. Default is 'test'.
    * `visualize`: Valid if `phase` is set to test. Output the attention maps on the original image. Set flag to `no-visualize` to test without visualizing. 
    * `load-model`: Load model from `model-dir` or not.
    * `target-vocab-size`: Target vocabulary size. Default is = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z

- Input and output
    * `data-base-dir`: The base directory of the image path in `data-path`. If the image path in `data-path` is absolute path, set it to `/`.
    * `data-path`: The path containing data file names and labels. Format per line: `image_path characters`.
    * `model-dir`: The directory for saving and loading model parameters (structure is not stored). Default is 'train'.
    * `log-path`: The path to put log. Default is 'log.txt'
    * `output-dir`: The path to put visualization results if `visualize` is set to True. Default is 'results'.
    * `steps-per-checkpoint`: Checkpointing (print perplexity, save model) per how many steps. Default is 500. 
    * `augmentation`: P(data augmentation). Default is 0.2. 

- Optimization
    * `num-epoch`: The number of whole data passes. Default is 1000. 
    * `batch-size`: Batch size. Only valid if `phase` is set to train. Default is 64.
    * `initial-learning-rate`: Initial (AdaDelta) learning rate. Default is 1. 

- Network
    * `reg-val`: Lambda for L2 regularization losses. Default is 0. 
    * `clip-gradients`: Whether to perform gradient clipping. Default is 'True'.
    * `max-gradient-norm`: Clip gradients to this norm. Default is 5. 
    * `target-embedding-size`: Embedding dimension for each target. Default is 10. 
    * `opt-attn`: Which attention mechanism to use: 'softmax' (default); 'log_softmax'; 'sigmoid'; 'no_attn'.
    * `use-gru`: Use GRU for decoder (rather than LSTM). Default is 'True'.
    * `attn-num-hidden`: Number of hidden units in attention decoder cell. Default is 128. 
    * `attn-num-layers`: Number of layers in attention decoder cell. Default is 2. (Encoder number of hidden units will be `attn-num-hidden`*`attn-num-layers`).