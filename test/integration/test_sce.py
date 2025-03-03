import numpy as np
from flax import linen as nn

from turbanet import TurbaTrainState, softmax_cross_entropy


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
input_size = 3
output_size = 2

# Training Parameters
learning_rate = 1e-2
dataset_size = 8
bootstrap_size = 8
epochs = 1000

# Generate a single batch input/output
input = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
)
output = np.array([[0], [1], [1], [1], [0], [1], [1], [0]])
output = np.eye(output_size)[output]

# # Setup batch
data = {
    "input": np.array(
        [
            input[np.random.choice(dataset_size, bootstrap_size, replace=False)]
            for _ in range(swarm_size)
        ]
    ),
    "output": np.array(
        [
            output[np.random.choice(dataset_size, bootstrap_size, replace=False)]
            for _ in range(swarm_size)
        ]
    ),
}

# Create networks
my_states1 = TurbaTrainState.swarm(
    Brain(hidden_layers=1, hidden_size=8, output_size=output_size),
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
