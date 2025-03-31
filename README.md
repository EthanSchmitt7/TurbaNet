# TurbaNet

TurbaNet is a lightweight and user-friendly API wrapper for the JAX library, designed to simplify and accelerate the setup of swarm-based training, evaluation, and simulation of small neural networks.​ Based on the work presented by Will Whitney in his blog post from 2021.[^1]

## Key Features

- Simplified API: Provides an intuitive interface for configuring and managing swarm-based neural network tasks.​
- Efficiency: Leverages JAX's capabilities to offer accelerated computation for training and evaluation processes.​
- Flexibility: Supports various configurations, allowing users to tailor the swarm behavior to specific needs.​

## Installation

To install TurbaNet, ensure that you have Python and pip installed. Then, run:

`pip install turbanet`

## Getting Started

Here's a basic example demonstrating how to initialize and use TurbaNet:

```
import turbanet as tn

# Initialize the swarm with desired parameters
swarm = tn.Swarm(num_networks=10, input_size=784, hidden_layers=[128, 64], output_size=10)

# Train the swarm on your dataset
swarm.train(data_loader, epochs=10, learning_rate=0.01)

# Evaluate the swarm's performance
accuracy = swarm.evaluate(test_loader)
print(f"Swarm Accuracy: {accuracy}%")
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
