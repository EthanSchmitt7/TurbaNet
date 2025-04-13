import numpy as np
import optax
from flax import linen as nn

from turbanet import TurbaTrainState, l2_loss


class Brain(nn.Module):
    hidden_layers: int = 1
    hidden_size: int = 8
    output_size: int = 1

    @nn.compact
    def __call__(self, x):  # noqa ANN001
        for layer in range(self.hidden_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        x = nn.sigmoid(x)
        return x


# Network Parameters
swarm_size = 10
input_size = 9
output_size = 1

# Training Parameters
learning_rate = 1e-3
dataset_size = 10
epochs = 10000

# Generate a single batch input/output
input = np.random.rand(dataset_size, input_size)
output = (np.random.rand(dataset_size, output_size) > 0.5) * 1.0

# Setup batch
data = {
    "input": np.repeat(np.expand_dims(input, 0), swarm_size, axis=0),
    "output": np.repeat(np.expand_dims(output, 0), swarm_size, axis=0),
}

# Create networks
optimizer = optax.adam(learning_rate=learning_rate)
my_states1 = TurbaTrainState.swarm(
    Brain(hidden_layers=1, hidden_size=8, output_size=output_size),
    swarm_size=swarm_size,
    optimizer=optimizer,
    sample_input=input[0].reshape(1, input_size),
)

# Original loss
initial_loss, initial_prediction = my_states1.evaluate(data["input"], data["output"], l2_loss)
print(
    f"Initial loss\n"
    f"Mean: {np.mean(initial_loss):7.5} | "
    f"Min: {np.min(initial_loss):7.5} | "
    f"Max: {np.max(initial_loss):7.5}\n"
)

# Train for a while
for i in range(epochs):
    my_states1, loss, prediction = my_states1.train(data["input"], data["output"], l2_loss)

    if i % 100 == 0:
        print(
            f"Epoch {i:>5} | "
            f"Mean loss: {np.mean(loss):7.5} | "
            f"Min loss: {np.min(loss):7.5} | "
            f"Max loss: {np.max(loss):7.5}"
        )

assert np.mean(loss) < np.mean(initial_loss)
