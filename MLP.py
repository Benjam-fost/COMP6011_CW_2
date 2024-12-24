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
learning_rates = np.linspace(0.001, 5, 50)
optimal_learning_rate = None
best_accuracy = 0.00
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

# Calculate accuracy on test set
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(np.argmax(y_true, axis=1) == y_pred)
    return correct / y_true.shape[0]

train_errors = []
val_errors = []
accuracies = []
### Iterate 2000 times over each learning rate to find the optimal one ###
for lr in learning_rates:
    # Create and train a model for each learning rate
    model = MLP(input_size, hidden_size, output_size, lr)
    model.train_with_cross_validation(X_train, y_train, kfolds, epochs, log=False)
    # Append final error rates after 2000 epochs, per learning rate
    train_errors.append(model.train_errors[-1])
    val_errors.append(model.val_errors[-1])
    
    # Predict the test data
    y_test_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, y_test_pred)
    accuracies.append(accuracy)
    #print(f"Learning rate: {lr}\n  Test accuracy: {accuracy * 100:.2f}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        optimal_learning_rate = lr

# Get the optimal learning rate via getting the lowest test error
train_errors = np.array(train_errors)
val_errors = np.array(val_errors)
index = np.argmin(val_errors)
found_optimal_learning_rate = learning_rates[index]
found_accuracy = accuracies[index]
lowest_error = val_errors[index]
print(f"||| Optimal learning rate by validation error: {found_optimal_learning_rate:.4f}, Test accuracy: {found_accuracy * 100:.2f}% |||")
print(f"\n||| Optimal learning rate by accuracy: {optimal_learning_rate:.4f}, Test accuracy: {best_accuracy * 100:.2f}% |||\n")
print(f"Seed: {seed}")

### Plot train/test errors vs learning rate ###
plt.figure(figsize=(8,5))
plt.plot(learning_rates, train_errors, marker='o', label='Training Error')
plt.plot(learning_rates, val_errors, marker='o', label='Validation Error')
plt.axvline(x=found_optimal_learning_rate, color='g', linestyle='--', label=f"Lowest validation error ({lowest_error:.2f})")
plt.xlabel('Learning Rate')
plt.ylabel('Error rates')
plt.title('Training and Validation Error (2000 epochs) Rate per Learning Rate')
plt.legend()
plt.show()