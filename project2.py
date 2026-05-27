  # Project 2 - Data Classification Using AI
# DecodeLabs Artificial Intelligence Internship
# %%
# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# Features and labels
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create classification model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict using test data
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

# Output results
print("Predictions:", predictions)
print("Actual Values:", y_test)
print("Model Accuracy:", accuracy * 100, "%") 
# %%
