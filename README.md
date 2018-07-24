# pytorch_style_transfer

Built mostly based on the pytorch [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) example for artistic style transfer, this repository further adds some *Conditional Instance Normalization* layers to train transfer network with multiple style images at the same time. 

*Conditional Instance Normalization* was introduced in [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629). The creation and usage of such idea in this repository is based on my limit knowledge in python, pytorch and neural network. Other implementations:
* [Joel Moniz Lasagne and Theano implementation]https://github.com/joelmoniz/gogh-figure
* [Google Magenta TensorFlow implementation](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization)

## Usage

Please refer to [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) for more details. We only discuss 

Train
```
python neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1 --batch-size 4
```
*`--style-image`: the code will grab all files under the path as style images
*`--batch-size`: number of images fed in each batch, does not need to be equal to the number of style images


Stylize 
```
python neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0 --style-num 19 --style-id 18
```