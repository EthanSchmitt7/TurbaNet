import numpy as np
from flax import linen as nn

from turbanet import TurbaTrainState, softmax_cross_entropy


def binary_generator(bit_count: int) -> np.ndarray:
    binary_strings = []

    def genbin(n: int, bs: str = "") -> None:
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return np.array([[int(c) for c in s] for s in binary_strings])


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
swarm_size = 100
input_size = 6
output_size = 2

# Training Parameters
learning_rate = 1e-5
epochs = 100000

# Generate a single batch input/output
input = binary_generator(input_size)
output = np.eye(output_size)[np.random.randint(0, output_size, input.shape[0])]

# Setup batch
data = {
    "input": np.repeat(np.expand_dims(input, 0), swarm_size, axis=0),
    "output": np.repeat(np.expand_dims(output, 0), swarm_size, axis=0),
}


# Create networks
my_states1 = TurbaTrainState.swarm(
    Brain(hidden_layers=2, hidden_size=8, output_size=output_size),
    swarm_size=swarm_size,
    input_size=input_size,
    learning_rate=learning_rate,
)

# Original loss
# Take the mean of their answers instead
loss, prediction = my_states1.check_loss(data["input"], data["output"], softmax_cross_entropy)
print(f"Initial loss | Mean: {np.mean(loss)} | Min: {np.min(loss)} | Max: {np.max(loss)}\n")


# Train for a while
for i in range(epochs):
    my_states1, prediction, loss = my_states1.train(
        data["input"], data["output"], softmax_cross_entropy
    )

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
    np.expand_dims(input, 0), np.expand_dims(output, 0), softmax_cross_entropy
)
print(f"\nMean of Weights: {loss[0]}")


# Take the mean of their answers instead
loss, prediction = my_states1.check_loss(data["input"], data["output"], softmax_cross_entropy)
print("Mean of Solutions: ", np.mean(loss))
