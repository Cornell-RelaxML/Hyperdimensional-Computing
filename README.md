## Understanding Hyperdimensional Computing for Parallel Single-Pass Learning

#### Authors:
* [Tao Yu](http://www.cs.cornell.edu/~tyu/)*
* [Yichi Zhang](https://ychzhang.github.io)*
* [Zhiru Zhang](https://www.csl.cornell.edu/~zhiruz/)
* [Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)

*: Equal Contribution

### Introduction
This repo contains implementation of the group VSA and binary HDC model with random Fourier feature (RFF) encoding, described in the paper Understanding Hyperdimensional Computing for Parallel Single-Pass Learning.

Our RFF method and group VSA can outperform the state-of-the-art HDC model while maintaining hardware efficiency. For example, on MNIST,

Model | 1-Epoch Accuracy | 10-Epoch Accuracy | Circuit-Depth Complexity 
:----:|:----------------:|:-----------------:|:----------------------:|
Percep. | 94.3 % | 94.3 % | 1299
SOTA HDC | NA | 89.0 % | 295
RFF HDC | 95.4 % | 95.4 % | 295
RFF G(2^3)-VSA | 96.3 % | 95.7 % | 405

### Dependencies and Data
Numpy and PyTorch>=1.0.0 are required to run the implementation. Supported datasets include [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [ISOLET](https://archive.ics.uci.edu/ml/datasets/isolet) and [UCI-HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). We provide the ISOLET and UCI-HAR data in `dataset` folder.

### Usage
Please create the `./encoded_data` folder before running the following code.
```
$ python main.py [-h] [-lr LR] [-gamma GAMMA] [-epoch EPOCH] [-gorder GORDER] [-dim DIM] 
[-data_dir DATA_DIR] [-model MODEL]
optional arguments:
  -h, --help            show this help message and exit
  -lr LR                learning rate for optimizing class representative
  -gamma GAMMA          kernel parameter for computing covariance
  -epoch EPOCH          epochs of training
  -gorder GORDER        order of the cyclic group required for G-VSA
  -dim DIM              dimension of hypervectors
  -resume               resume from existing encoded hypervectors
  -data_dir DATA_DIR    Directory used to save encoded data (hypervectors)
  -dataset {mnist,fmnist,cifar,isolet,ucihar}
                        dataset (mnist | fmnist | cifar | isolet | ucihar)
  -raw_data_dir RAW_DATA_DIR
                        Raw data directory to the dataset
  -model {rff-hdc,linear-hdc,rff-gvsa}
                        feature and model to use: (rff-hdc | linear-hdc | rff-gvsa)
```
For example, 
```
$ python main.py -gamma 0.3 -epoch 10 -gorder 8 -dim 10000 -dataset mnist -model rff-gvsa
```

### Citation
If you find this repo useful, please cite:
```

```
