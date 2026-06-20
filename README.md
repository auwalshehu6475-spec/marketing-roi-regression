# ==========================================
# STEP 2: TRAIN-TEST SPLIT & FEATURE SCALING
# ==========================================
print("\n--- Step 2: Split and Scale Data ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ⚠️ FIX: Scaling one-hot encoded data is optional
# BEST PRACTICE: skip scaling OR use MinMaxScaler

# Option A (recommended): no scaling
X_train_scaled = X_train
X_test_scaled = X_test

# ==========================================
# STEP 3: KNN MODEL
# ==========================================

k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='hamming')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]

print("\nBest K:", best_k)

# ==========================================
# STEP 4: FINAL MODEL
# ==========================================

final_model = KNeighborsClassifier(n_neighbors=best_k, metric='hamming')
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)

# ==========================================
# STEP 5: EVALUATION
# ==========================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))