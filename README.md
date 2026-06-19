from sklearn.pipeline import Pipeline

k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"k = {k} | CV Accuracy: {scores.mean():.4f}")