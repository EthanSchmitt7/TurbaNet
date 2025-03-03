import numpy as np
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
        x = nn.softmax(x)
        return x


# Network Parameters
swarm_size = 1
input_size = 3
output_size = 1

# Training Parameters
learning_rate = 1e-6
dataset_size = 8
bootstrap_size = 8
epochs = 1000

# Generate a single batch input/output
input = np.random.rand(dataset_size, input_size)
output = np.random.rand(dataset_size, output_size)

# # Setup batch
data = {
    "input": np.repeat(input, swarm_size).reshape(swarm_size, input.shape[0], input.shape[1]),
    "output": np.repeat(output, swarm_size).reshape(swarm_size, output.shape[0], output.shape[1]),
}

# Create networks
my_states1 = TurbaTrainState.swarm(
    Brain(hidden_layers=1, hidden_size=8, output_size=output_size),
    swarm_size=swarm_size,
    input_size=input_size,
    learning_rate=learning_rate,
)

# Original loss
loss, prediction = my_states1.check_loss(data["input"], data["output"], l2_loss)
print(f"Initial loss | Mean: {np.mean(loss)} | Min: {np.min(loss)} | Max: {np.max(loss)}\n")


# Train for a while
for i in range(epochs):
    my_states1, prediction, loss = my_states1.train(data["input"], data["output"], l2_loss)

    if i % 100 == 0:
        print(
            f"Epoch {i} | "
            f"Mean loss: {np.mean(loss)} | "
            f"Min loss: {np.min(loss)} | "
            f"Max loss: {np.max(loss)}"
        )

print(f"\nFinal loss | Mean: {np.mean(loss)} | Min: {np.min(loss)} | Max: {np.max(loss)}")

# Merge the swarm and check the loss
my_states2: TurbaTrainState = my_states1.merge()
my_states2, prediction, loss = my_states2.train(
    np.expand_dims(input, 0), np.expand_dims(output, 0), l2_loss
)
print(f"\nMean of Weights: {loss[0]}")


# Take the mean of their answers instead
loss, prediction = my_states1.check_loss(data["input"], data["output"], l2_loss)
print("Mean of Solutions: ", np.mean(loss))

# Take the mean of their answers instead
loss, prediction = my_states1.check_loss(data["input"], data["output"], l2_loss)
print("Mean of Solutions: ", np.mean(loss))
