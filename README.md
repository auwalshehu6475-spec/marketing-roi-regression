# knn_project.py
# K-Nearest Neighbor Classification - Pattern Recognition Mini Project

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def run_project():
    print("====================================================")
    print("  K-Nearest Neighbor Pattern Recognition Project    ")
    print("====================================================\n")
    
    # 1. Load and Explore Dataset
    if not os.path.exists('Iris.csv'):
        print("Error: Iris.csv not found in the current directory.")
        return
        
    df = pd.read_csv('Iris.csv')
    print("--- 1. Dataset Head ---")
    print(df.head())
    
    print("\n--- Dataset Summary Statistics ---")
    print(df.describe())
    
    # Drop 'Id' column if it exists
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    # Check for missing values
    print("\n--- Checking for Missing Values ---")
    missing = df.isnull().sum()
    print(missing)
    
    # 2. Visualize Feature Distributions
    print("\n--- 2. Visualizing Feature Distributions ---")
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=feature, hue='Species', kde=True, element='step', stat='density', common_norm=False)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    print("Saved feature distribution plot to 'feature_distributions.png'")
    
    # 3. Split Data into Training and Testing Sets (80/20 split)
    print("\n--- 3. Splitting Dataset (80% Train, 20% Test) ---")
    X = df[features]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training Features Shape: {X_train.shape} | Labels Shape: {y_train.shape}")
    print(f"Testing Features Shape:  {X_test.shape}  | Labels Shape: {y_test.shape}")
    
    # 4. Feature Scaling (Standardization)
    print("\n--- 4. Applying StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features normalized successfully.")
    
    # 5. Initialize and Train Initial KNN Model
    initial_k = 3
    print(f"\n--- 5. Training Initial KNN Classifier (k={initial_k}) ---")
    knn_initial = KNeighborsClassifier(n_neighbors=initial_k)
    knn_initial.fit(X_train_scaled, y_train)
    y_pred_initial = knn_initial.predict(X_test_scaled)
    print(f"Initial Model Accuracy on Test Set: {accuracy_score(y_test, y_pred_initial):.4f}")
    
    # 6. Hyperparameter Tuning using Cross-Validation
    print("\n--- 6. Testing different k values using 5-Fold Cross-Validation ---")
    k_values = [1, 3, 5, 7, 9, 11]
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"k = {k:2d} | Mean CV Accuracy = {scores.mean():.4f}")
        
    # 7. Plot k vs. Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('k-NN: Hyperparameter Tuning via Cross-Validation', fontsize=12)
    plt.xlabel('Number of Neighbors (k)', fontsize=10)
    plt.ylabel('Mean Cross-Validation Accuracy', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    plt.savefig('k_vs_accuracy.png')
    plt.close()
    print("Saved Hyperparameter Tuning plot to 'k_vs_accuracy.png'")
    
    # Identify the optimal k
    optimal_k = k_values[np.argmax(cv_scores)]
    print(f"Optimal k identified based on CV: {optimal_k}")
    
    # 8. Evaluate Final Model
    print(f"\n--- 8. Training and Evaluating Final Model (k={optimal_k}) ---")
    knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_final.fit(X_train_scaled, y_train)
    y_pred_final = knn_final.predict(X_test_scaled)
    
    print(f"Final Model Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final))
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_final, labels=knn_final.classes_)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_final.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')
    plt.title(f'Confusion Matrix (k={optimal_k})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix to 'confusion_matrix.png'")
    
    # 9. Test Predictions on New Sample Data
    print("\n--- 9. Testing Predictions on New Sample Points ---")
    new_samples = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2],  # Expected: Iris-setosa
        [6.5, 3.0, 4.5, 1.5],  # Expected: Iris-versicolor
        [7.3, 2.9, 6.3, 1.8]   # Expected: Iris-virginica
    ], columns=features)
    
    new_samples_scaled = scaler.transform(new_samples)
    new_predictions = knn_final.predict(new_samples_scaled)
    
    for i, (sample, pred) in enumerate(zip(new_samples.values, new_predictions), 1):
        print(f"Sample {i} {list(sample)} -> Predicted Class: {pred}")
        
    print("\n====================================================")
    print("              Project Completed Successfully        ")
    print("====================================================")

if __name__ == "__main__":
    run_project()
