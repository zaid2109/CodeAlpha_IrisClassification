import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and explore the dataset
print("=== Iris Dataset Classification ===\n")

# Load the dataset
df = pd.read_csv('Iris.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Distribution:")
print(df['Species'].value_counts())

# Data preprocessing
print("\n=== Data Preprocessing ===")

# Remove the Id column as it's not needed for classification
df = df.drop('Id', axis=1)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nClasses: {le.classes_}")
print(f"Encoded labels: {np.unique(y_encoded)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# Data visualization
print("\n=== Data Visualization ===")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Feature Analysis', fontsize=16)

# Pairwise relationships
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', ax=axes[0,0])
axes[0,0].set_title('Sepal Length vs Sepal Width')

sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', ax=axes[0,1])
axes[0,1].set_title('Petal Length vs Petal Width')

# Box plots
sns.boxplot(data=df, x='Species', y='SepalLengthCm', ax=axes[1,0])
axes[1,0].set_title('Sepal Length by Species')

sns.boxplot(data=df, x='Species', y='PetalLengthCm', ax=axes[1,1])
axes[1,1].set_title('Petal Length by Species')

plt.tight_layout()
plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop('Species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as 'iris_analysis.png' and 'correlation_heatmap.png'")

# Machine Learning Models
print("\n=== Machine Learning Model Training ===")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f}")

# Display results comparison
print("\n=== Model Performance Comparison ===")
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df)

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']
print(f"\nBest performing model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Detailed evaluation of the best model
print(f"\n=== Detailed Evaluation - {best_model_name} ===")

# Get the best model
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=le.classes_))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance (for Random Forest)
if best_model_name == 'Random Forest':
    print("\n=== Feature Importance ===")
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance - Random Forest')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Model comparison visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for i, acc in enumerate(results_df['Accuracy']):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Prediction Examples ===")
# Show some prediction examples
test_examples = X_test.head(5)
test_predictions = best_model.predict(X_test_scaled[:5])
test_actual = y_test[:5]

print("\nTest Examples (First 5):")
for i in range(5):
    actual_species = le.classes_[test_actual[i]]
    predicted_species = le.classes_[test_predictions[i]]
    correct = "✓" if actual_species == predicted_species else "✗"
    
    print(f"\nExample {i+1}:")
    print(f"  Features: SepalLength={test_examples.iloc[i]['SepalLengthCm']:.1f}, "
          f"SepalWidth={test_examples.iloc[i]['SepalWidthCm']:.1f}, "
          f"PetalLength={test_examples.iloc[i]['PetalLengthCm']:.1f}, "
          f"PetalWidth={test_examples.iloc[i]['PetalWidthCm']:.1f}")
    print(f"  Actual: {actual_species}")
    print(f"  Predicted: {predicted_species} {correct}")

print(f"\n=== Summary ===")
print(f"- Dataset: {df.shape[0]} samples, {df.shape[1]-1} features, 3 classes")
print(f"- Best model: {best_model_name}")
print(f"- Test accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"- Classes: {', '.join(le.classes_)}")
print("\nFiles generated:")
print("- iris_analysis.png: Data visualization plots")
print("- correlation_heatmap.png: Feature correlation matrix")
print("- confusion_matrix.png: Confusion matrix for best model")
if best_model_name == 'Random Forest':
    print("- feature_importance.png: Feature importance plot")
print("- model_comparison.png: Accuracy comparison of all models")
