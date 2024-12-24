import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

# Config
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
    ]
# Used for reproducibility
seed = np.random.randint(0, 999)
# Optimal learning rate
learning_rate = 0.2357
# Optimal number of epochs
epochs = 2500
# Number of neurons in the hidden layer
hidden_size = 4
# Number of folds for cross-validation
kfolds = 5

# Get the dataset
penguins = sns.load_dataset("penguins")
# Sanitise the data, removing all incomplete entries
penguins.dropna(inplace=True)

# Features
X = penguins[features].values
# Features normalisation
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Map of one-hot encoded species to species names
species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
# Labels
y = penguins["species"].map(species_map).values
# One-hot encoding
y_one_hot = np.eye(3)[y]

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Add small epsilon to avoid log(0)

# MLP class with sigmoid activation for both layers
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))

        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)  # Hidden layer
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)  # Output layer
        return self.output

    def backward(self, X, Y):
        # Output layer error
        output_error = self.output - Y
        output_delta = output_error * sigmoid_derivative(self.output)

        # Hidden layer error
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights2 -= np.dot(self.hidden.T, output_delta) * self.learning_rate
        self.bias2 -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights1 -= np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias1 -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train_with_cross_validation(self, X, y, kfolds, epochs, log = False):
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
        self.train_errors = np.zeros(epochs)
        self.val_errors = np.zeros(epochs)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if log: print(f"Training Fold {fold + 1}/{kfolds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train for each epoch
            for epoch in range(epochs):
                y_pred = self.forward(X_train)
                self.backward(X_train, y_train)

                # Calculate training loss
                train_loss = categorical_cross_entropy(y_train, y_pred)
                self.train_errors[epoch] += train_loss / kfolds

                # Calculate validation loss
                y_val_pred = self.forward(X_val)
                val_loss = categorical_cross_entropy(y_val, y_val_pred)
                self.val_errors[epoch] += val_loss / kfolds

            if log: print(f"Fold {fold + 1} completed.")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Calculate accuracy on test set
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(np.argmax(y_true, axis=1) == y_pred)
    return correct / y_true.shape[0]

# Metrics
splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
accuracies = []
# Iterate, at 2000 epochs and LR of 0.1030, over each train/test split ratio to find the optimal one
for test_ratio in splits:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_one_hot,
        test_size=test_ratio,
        random_state=seed
    )
    input_size = X_train.shape[1]  # 4 features per sample
    output_size = y_train.shape[1]  # 3 classes
    # Create and train a model for each split ratio
    model = MLP(input_size, hidden_size, output_size, learning_rate)
    model.train_with_cross_validation(X_train, y_train, kfolds, epochs, log=False)
    # Get predictions at split ratio, after 2000 epochs           
    y_test_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, y_test_pred)
    accuracies.append(accuracy)
    # Output metrics for data table in report
    print(f"||| Test/Train Ratio: {int(test_ratio * 10)}:{int(10 - test_ratio * 10)} | Accuracy: {accuracy * 100:.2f}% |||")

print(f"Seed: {seed}")

ticks = [int(ratio * 100) for ratio in splits]
### Plot train/test errors vs learning rate ###
plt.figure(figsize=(10,6))
plt.bar(
    ticks,
    [int(accuracy * 100) for accuracy in accuracies],
    color = 'forestgreen',
    edgecolor = 'black',
    width = 6
)
plt.plot(
    ticks,
    [int(accuracy * 100) for accuracy in accuracies],
    color='black',
    linestyle='--',
    label="Accuracy Trend Line"
)
plt.xlabel('Test Ratio (%)')
plt.xticks(ticks)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Test/Train Ratio')
plt.show()