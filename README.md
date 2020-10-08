# Scanned Receipt OCR by Convolutional-Recurrent Neural Network

This is a `pytorch` implementation of CRNN, which is based on @meijieru's repository [here](https://github.com/meijieru/crnn.pytorch). It was re-implmented and modified to work with the problem set given by Egregore.

## Introduction


We applied a modified CRNN in this task( task 2 for the Universal Text Extractor solution from scanned images). CRNN is a conventional scene text recognition method including convolutional layers, bidirectional LSTM layers, and a transcription layer in sequence. 


In scanned receipts each text usually contains several words. We add the blank space between words to the alphabet for LSTM prediction and thus improve the network from single word recognition to multiple words recognition. Moreover, we double the input image width to tackle the overlap problem of long texts after max-pooling and stack one more LSTM, enhancing the accuracy per character in the training set from 62% to 83%.

Testing is done with SROIE dataset

## Dependency

1. [warp-ctc-pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
2. lmdb
3. Python 3.6+ and pyTorch

## Prediction

As is shown in the introduction image, CRNN only accepts image that has single line of word. Therefore, we must provide bounding boxes for the whole receipt image.

1. Put image under folder `./data_test/` and bounding box text file under `./boundingbox/`, the name of image file and text file must correspond.

2. Download pre-trained model from [here](https://drive.google.com/open?id=1X3_pNnLNEdwEcgiFrtwvc4uYXzkZ9Zjw) and put the weight file under `./expr/` folder 

3. To predict, just run `python main.py`. You can change the code inside to visualise output or prepare result for task 3.

example result:
```
tan chay yee
81750 masai johor
sales persor : fatin
tax invoice
total inclusive gst:
invoice no : pegiv1030765
email: ng@ojcgroup.com
bill to"
date
description
address
total:
cashier
:the peak quarry works
```

## Training

Training a CRNN requires converting data into `lmdb` format.

1. Divide training data into training and validating dataset, put each portion into `./data_train/` and `./data_valid`. An example could be found [here](https://drive.google.com/open?id=1JKLh7Jq1VXVNW1InKJv6xUrc21zCQNpE)

2. Run `create_dataset.py` to create lmdb dataset. The created dataset could be found inside `./dataset`

3. After preparing dataset, just run:
   ```shell
   python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda
   ```
   with desired options
   Trapti's Note: runs with : python train.py --adadelta --trainroot dataset/train --valroot dataset/val --cuda

4. Trained model output will be in `./expr/`
