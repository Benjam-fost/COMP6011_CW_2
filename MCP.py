import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
learning_rate = 0.1030
# Optimal number of epochs
epochs = 2000
optimal_learning_rate = None
best_accuracy = 0.00

# Get the dataset
penguins = sns.load_dataset("penguins")
# Sanitise the data, removing all incomplete entries
penguins.dropna(inplace=True)

# Features
X = penguins[features].values

# Map of one-hot encoded species to species names
species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

# One-hot encoding
## Labels
y = penguins["species"].map(species_map).values
# One-hot encoding
y_one_hot = np.eye(3)[y]

# Features normalisation
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Add small epsilon (1e-9) to avoid log(0)

class MulticlassPerceptron:
    def __init__(self, input_size, output_size, learning_rate):

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate


        self.weights = np.random.randn(input_size, output_size) * 0.01 # Initialise weights randomly with small values
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        #Forward propagation
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias  # Linear combination
        self.output = sigmoid(self.z)  # Sigmoid activation
        return self.output

    def backward(self, X, Y):
        #Backward propagation (gradient calculation and weight update)
        output_error = self.output - Y  # Error term (difference from true values)
        output_delta = output_error * sigmoid_derivative(self.output)  # Gradient at the output

        self.weights -= np.dot(self.input.T, output_delta) * self.learning_rate  # Weight update
        self.bias -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate  # Bias update

    def train(self, X_train, y_train, X_test, y_test, epochs, log=False):
        #Train the perceptron using backpropagation
        self.train_errors = []
        self.test_errors = []

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)
            # Backward pass
            self.backward(X_train, y_train)
            
            # Calculate training error (loss)
            train_loss = categorical_cross_entropy(y_train, y_pred)
            self.train_errors.append(train_loss)

            # Calculate test error (loss) at every epoch for evaluation
            y_test_pred = self.forward(X_test)
            test_loss = categorical_cross_entropy(y_test, y_test_pred)
            self.test_errors.append(test_loss)

            # Output log of process
            if log and epoch % 100 == 0:
                print(f"|| Epoch {epoch} ||\n  Learning rate: {self.learning_rate}\n  Training Loss: {train_loss}\n  Test Loss: {test_loss}")

    def predict(self, X):
        #Predict the class for given input
        output = self.forward(X)
        return np.argmax(output, axis=1)  # Return class with the highest probability

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy."""
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
    model = MulticlassPerceptron(input_size, output_size, learning_rate)
    model.train(X_train, y_train, X_test, y_test, epochs, log=False)
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