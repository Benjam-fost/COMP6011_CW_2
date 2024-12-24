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
learning_rate = 0.001
# Optimal number of epochs
epochs = 5000
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

# Split data into optimal training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=seed)
input_size = X_train.shape[1]
output_size = y_train.shape[1]

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

# Initialize and train the MLP model with one hidden layer

# Create and train the MLP model
model = MLP(input_size, hidden_size, output_size, learning_rate)

# Train the model using K-Fold Cross-Validation
model.train_with_cross_validation(X_train, y_train, kfolds, epochs)

# Calculate accuracy on test set
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(np.argmax(y_true, axis=1) == y_pred)
    return correct / y_true.shape[0]

y_test_pred = model.predict(X_test)
accuracy = calculate_accuracy(y_test, y_test_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
print(f"Seed: {seed}")
# Plot training and validation error curves
plt.plot(model.train_errors, label='Training Error')
plt.plot(model.val_errors, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error Rates')
plt.legend()
plt.title('Training and Validation Error Curves')
plt.show()
