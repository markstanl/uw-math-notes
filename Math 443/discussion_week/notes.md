https://poloclub.github.io/cnn-explainer/

# CNNs
Convolutional Neural Networks are a special type of neural network that works especially well for image classification.
CNNs are quite similar to MLPs, but they have a few differences, specifically in the hidden layers. CNNs are made up of
the following layers: input layer, convolutional layer, pooling layer, flattening layer, dense layer, and output layer. The Convolutional layer
adds filters over the input image to detect feature patters. The pooling layer reduces the size of the resulting feature maps to 
reduce computation time. The dense layer is a fully connected layer that takes the output of the convolutional and pooling layers.
The network trains via backpropagation and gradient descent.

## Convolutional Layer
An image can be represented as a tensor, with dimensions height, width, as well as 3 channels for RGB. The convolutional
layer has the term 'kernel', which is just their term for the learned weights. These kernels extract features on what
distinguishes each type of image from another. Another term 'feature' refers to the larger, more abstract
concepts of what makes 1 classification different from another (as opposed to the traditional pixel values). 

For each unique kernel function, the intermediate will apply that to each channel of the input image. This 
happens via taking the dot product of the kernel and the input image. You can imagine, though, that this kernel function
be applied through a sliding window over the entire image. This sliding window is called the 'stride'. Each stride is
summed together with a bias, and that is the next neuron, which is passed through an activation function.

The size of the stride and kernel are hyperparameters and can be tuned. The stride is how many pixels the kernel moves.

It is important to note that the each input channel has its own kernel. Furthermore, the kernel itself is a matrix, which
is why the dot product is used. (A 3x3 matrix for a single channel multiplied by the 3x3 input image matrix returns a single value).

## Pooling Layer
The pooling layer serves the purpose of reducing size of the final feature maps from the convolutional layers.
Max-Pooling in one such method, which takes kernel and stride hyperparameters. This simply performs a max operation on 
the matrix of size kernel, and moves it by the stride. This does a good job of reducing the size of the feature maps, 
while still keeping largely important features, and reduces overfitting.

## Flattening Layer
This layer simply converts the 2D matrix of the final pooling layer into a 1D vector. 
This is necessary for the dense layer.

## Dense Layer
This layer is a fully connected layer that takes the output of the convolutional and pooling layers, as the name 
suggests. This layer is the same as the hidden layers in a MLP. The dense layer is responsible for the final classification.

