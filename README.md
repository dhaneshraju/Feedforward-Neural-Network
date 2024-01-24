# Feedforward Neural Network with Backpropagation
## Overview
This repository contains the implementation of a Feedforward Neural Network (FNN) with Backpropagation, a fundamental type of artificial neural network where information flows in one direction during the forward pass and errors are propagated backward during the backpropagation phase for training.

## Features
Modularity: The network architecture is designed to be easily customizable. You can modify the number of layers, neurons in each layer, and activation functions according to your specific needs.

Forward Propagation: The feedforward mechanism is implemented efficiently, ensuring fast and accurate predictions for given inputs.

Backpropagation: The network incorporates the backpropagation algorithm for training. During the training phase, errors are calculated, and the weights are updated to minimize these errors, allowing the network to learn from the provided data.

Activation Functions: Choose from a variety of activation functions such as ReLU, Sigmoid, or Hyperbolic Tangent to introduce non-linearity in the network.

Loss Function: The network supports different loss functions, allowing you to tailor the training process to your specific problem, whether it's regression or classification.

so if you just run the file it will automatically train the model and start to predict, Because ive allready implement and attached the data-set with it.

## Getting Started
Prerequisites
Ensure you have Python installed (version latest).
Install required dependencies by running:
Copy code
pip install -r requirements.txt
#### Usage
Clone the Repository:

#### bash
#### Copy code
git clone https://github.com/dhaneshraju/feedforward-neural-network.git

cd feedforward-neural-network

Train the Neural Network:

#### Copy code
python final_feed_forward_neural_network.py

Make Predictions:

#### Copy code
python predict.py

Customization

Feel free to experiment with the following parameters to customize the network architecture:

Number of layers
Number of neurons in each layer
Activation functions
Learning rate
Loss function
Optimizer
Explore the code in final_feed_forward_neural_network.py and adjust the hyperparameters accordingly.
Which the above code runs only for the Neural Network
I integrated this Nueral Network with the Lunar Landers game, i use this network to land the lander in the specific spot automatically without any human interventions
(5/20 attempts) Still working in the model.

The below image shows the example of the game:

[Main Page](/Main-FFN/Image-Source/1.png)


[Lunar Lander](/Main-FFN/Image-Source/2.png)


## Contributing
If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. We welcome contributions!
