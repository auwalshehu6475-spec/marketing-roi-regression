import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# ==========================================
# 1. Load and Prepare Data
# ==========================================
print("Loading data...")
df = pd.read_csv("4bb936ec-189a-4a96-8818-e19ac0d167e5.csv")

# Target variable encoding (satisfied = 1, neutral/dissatisfied = 0)
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Identify categorical and numerical columns
categorical_cols = ['Customer Type', 'Type of Travel', 'Class']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['satisfaction']]

# Separate features and target BEFORE any preprocessing
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# ==========================================
# 2. Train/Test Split (do this first!)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. Fit preprocessors ONLY on training data
# ==========================================
# Impute missing values with median (fitted on train, applied to both)
imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# Encode categorical variables (fit on train, transform test)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    # For test set, handle unseen labels by mapping them to a new category if needed,
    # or simply transform; here we assume no unseen categories appear.
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# ==========================================
# 4. Scale features (fit on train, transform both)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. Hyperparameter Tuning (GridSearchCV)
# ==========================================
print("Tuning Decision Tree...")
param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

dt_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)          # trees don't require scaling

best_dt = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# ==========================================
# 6. Model Evaluation & Comparison
# ==========================================
# Decision Tree Evaluation
y_pred_dt = best_dt.predict(X_test)
f1_dt = f1_score(y_test, y_pred_dt)

print("\n--- Decision Tree Performance ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print(f"F1-Score (Satisfied Class): {f1_dt:.4f}")

# Logistic Regression Baseline Comparison
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_scaled, y_train)
y_pred_lr = lr_clf.predict(X_test_scaled)
f1_lr = f1_score(y_test, y_pred_lr)

print("\n--- Logistic Regression Performance ---")
print(f"F1-Score (Satisfied Class): {f1_lr:.4f}")

# ==========================================
# 7. Feature Importance & Tree Visualization
# ==========================================
importances = best_dt.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,    # column order is unchanged from original X
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top operational drivers of satisfaction ---")
print(feature_importance_df.head(5))

# Plot Decision Tree (Pruned to depth 3 for readability)
plt.figure(figsize=(20,10))
plot_tree(best_dt, max_depth=3, feature_names=X.columns,
          class_names=['Dissatisfied', 'Satisfied'], filled=True, rounded=True)
plt.savefig('decision_tree_pathways.png', dpi=300)
print("\nTree visualization saved as 'decision_tree_pathways.png'")

# ==========================================
# 8. Save Artifacts for Deployment
# ==========================================
artifacts = {
    'model': best_dt,                # Decision Tree (no scaling needed)
    'encoders': label_encoders,      # fitted LabelEncoders
    'imputer': imputer,              # fitted median imputer
    'numerical_cols': numerical_cols, # needed to know which cols to impute
    'features': list(X.columns)       # exact column order expected by the model
}
joblib.dump(artifacts, "airline_satisfaction_model.pkl")
print("Model artifacts successfully saved!")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import warnings

app = FastAPI(title="Airline Customer Satisfaction API", version="1.0")

# Load artifacts
try:
    artifacts = joblib.load("airline_satisfaction_model.pkl")
    model = artifacts['model']
    encoders = artifacts['encoders']
    feature_names = artifacts['features']
    numerical_cols = artifacts['numerical_cols']   # now stored during training
    imputer = artifacts['imputer']
except FileNotFoundError:
    raise RuntimeError("Model file 'airline_satisfaction_model.pkl' not found. "
                       "Please run the training script first.")
except KeyError as e:
    raise RuntimeError(f"Missing key in artifacts: {e}. "
                       "The training script must save 'numerical_cols'.")

# Request schema (match the training feature names exactly)
class PassengerData(BaseModel):
    Customer_Type: str          # "Loyal Customer" / "disloyal Customer"
    Type_of_Travel: str         # "Personal Travel" / "Business travel"
    Class: str                  # "Eco", "Business", "Eco Plus"
    Age: int
    Flight_Distance: int
    Seat_comfort: int           # 0-5
    Departure_Arrival_time_convenient: int
    Food_and_drink: int
    Gate_location: int
    Inflight_wifi_service: int
    Inflight_entertainment: int
    Online_support: int
    Ease_Online_booking: int
    On_board_service: int
    Leg_room_service: int
    Baggage_handling: int
    Checkin_service: int
    Cleanliness: int
    Online_boarding: int
    Departure_Delay_in_Minutes: int
    Arrival_Delay_in_Minutes: float


@app.get("/")
def home():
    return {"message": "Airline Customer Satisfaction API is live. Use /predict to POST."}


@app.post("/predict")
def predict_satisfaction(passenger: PassengerData):
    try:
        # 1. Build a DataFrame with the exact column names used during training
        input_dict = {
            'Customer Type': [passenger.Customer_Type],
            'Type of Travel': [passenger.Type_of_Travel],
            'Class': [passenger.Class],
            'Age': [passenger.Age],
            'Flight Distance': [passenger.Flight_Distance],
            'Seat comfort': [passenger.Seat_comfort],
            'Departure/Arrival time convenient': [passenger.Departure_Arrival_time_convenient],
            'Food and drink': [passenger.Food_and_drink],
            'Gate location': [passenger.Gate_location],
            'Inflight wifi service': [passenger.Inflight_wifi_service],
            'Inflight entertainment': [passenger.Inflight_entertainment],
            'Online support': [passenger.Online_support],
            'Ease of Online booking': [passenger.Ease_Online_booking],
            'On-board service': [passenger.On_board_service],       # corrected hyphen
            'Leg room service': [passenger.Leg_room_service],
            'Baggage handling': [passenger.Baggage_handling],
            'Checkin service': [passenger.Checkin_service],
            'Cleanliness': [passenger.Cleanliness],
            'Online boarding': [passenger.Online_boarding],
            'Departure Delay in Minutes': [passenger.Departure_Delay_in_Minutes],
            'Arrival Delay in Minutes': [passenger.Arrival_Delay_in_Minutes]
        }
        df = pd.DataFrame(input_dict)

        # 2. Impute numerical columns using the exact same columns the imputer was fit on
        #    Validate that all required numerical columns are present
        missing_num_cols = set(numerical_cols) - set(df.columns)
        if missing_num_cols:
            raise ValueError(f"Missing numerical columns in input: {missing_num_cols}")
        df[numerical_cols] = imputer.transform(df[numerical_cols])

        # 3. Encode categorical columns with robust handling of unseen categories
        for col in ['Customer Type', 'Type of Travel', 'Class']:
            le = encoders[col]
            # Map input values to existing labels; raise error for unseen ones
            known_classes = set(le.classes_)
            if passenger.dict()[col] not in known_classes:
                raise ValueError(
                    f"Column '{col}' received unknown category '{passenger.dict()[col]}'. "
                    f"Expected one of: {list(le.classes_)}"
                )
            df[col] = le.transform(df[col])

        # 4. Enforce the exact column order expected by the model
        df = df[feature_names]

        # 5. Predict
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        # The model's classes_ are [0, 1] (0 = not satisfied, 1 = satisfied)
        probability_satisfied = proba[1]  # index 1 corresponds to satisfied

        return {
            "satisfaction_prediction": "Satisfied" if pred == 1 else "Neutral/Dissatisfied",
            "satisfaction_probability": round(float(probability_satisfied), 4)
        }

    except Exception as e:
        # Return 422 for predictable user input errors, 500 for unexpected ones.
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")