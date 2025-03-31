# TurbaNet

TurbaNet is a lightweight and user-friendly API wrapper for the JAX library, designed to simplify and accelerate the setup of swarm-based training, evaluation, and simulation of small neural networks.​ Based on the work presented by Will Whitney in his blog post from 2021.[^1]

## Key Features

- Simplified API: Provides an intuitive interface for configuring and managing swarm-based neural network tasks.​
- Efficiency: Leverages JAX's capabilities to offer accelerated computation for training and evaluation processes.​
- Flexibility: Supports various configurations, allowing users to tailor the swarm behavior to specific needs.​

## Installation

To install TurbaNet, ensure that you have Python and pip installed. Then, run:

`pip install turbanet`

TurbaNet train states require models and optimizers from Flax and Optax which can be installed with:

`pip install flax optax`

## Getting Started

Here's a basic example demonstrating how to initialize and use TurbaNet:

```
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from turbanet import TurbaTrainState, mse

# Set numpy random seed for reproducable results
np.random.seed(0)

# Sample input
X_data = np.random.randint(0, 2, (10, 10)).astype(float)
y_data = np.random.randint(0, 2, (10, 1)).astype(float)


# Define model for the swarm
class MyModel(nn.Module):
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x


# Define optimizer
optimizer = optax.adam(learning_rate=0.01)

# Define the size of the swarm
swarm_size = 10

# Initialize the swarm with desired parameters
swarm = TurbaTrainState.swarm(MyModel(), optimizer, swarm_size, X_data[0].reshape(1, -1))

# Train the swarm on your dataset
epochs = 100
losses = np.zeros((epochs, swarm_size))
for epoch in range(epochs):
    X = np.expand_dims(X_data, 0).repeat(len(swarm), axis=0)
    y = np.expand_dims(y_data, 0).repeat(len(swarm), axis=0)
    swarm, losses[epoch], predictions = swarm.train(X, y, mse)

# Plot the loss curves from training
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

```

For more detailed tutorials and examples, please refer to the documentation.

## Contributing

We welcome contributions to TurbaNet! If you'd like to contribute, please follow these steps:

    Fork the repository: Click the "Fork" button at the top right of the GitHub page.​

    Clone your fork:

    `git clone https://github.com/your-username/TurbaNet.git`

3. Create a new branch:

`git checkout -b feature/your-feature-name`

4. Make your changes: Implement your feature or fix the identified issue.​ 5. Commit your changes:

`git commit -m "Description of your changes"`

6. Push to your fork:

`git push origin feature/your-feature-name`

7. Submit a Pull Request: Navigate to the original repository and click on "New Pull Request" to submit your changes for review.​

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/EthanSchmitt7/TurbaNet/blob/main/LICENSE) file for more details.

## References
[^1]: Whitney, W. (2021). Parallelizing neural networks on one GPU with JAX. Will Whitney's Blog.
https://willwhitney.com/parallel-training-jax.html
