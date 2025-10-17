import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- 1. Dataset Handling ---
file_name = 'PulseBat Dataset.xlsx' 

try:
    print(f"Attempting to read file: {file_name} as an Excel file...")
    # 1. Try reading as Excel using the openpyxl engine
    df = pd.read_excel(file_name, engine='openpyxl')
    
except Exception as e:
    # 2. Fallback: If Excel reading fails (due to format or engine issue), try reading as CSV.
    # This addresses the likelihood that the content is CSV-formatted despite the .xlsx name.
    print(f"Excel read failed ({e}). Falling back to CSV reading.")
    try:
        df = pd.read_csv(file_name)
    except Exception as e_csv:
        print(f"CSV read also failed: {e_csv}. Please verify file name and format.")
        exit()

df.columns = df.columns.str.strip()

# Identify Features (U1-U21 cell voltages) and Target (SOH)
CELL_FEATURES = [f'U{i}' for i in range(1, 22)]
TARGET = 'SOH'

# Clean data by dropping rows with NaN values in relevant columns
df_clean = df.dropna(subset=CELL_FEATURES + [TARGET]).copy()
print(f"\nSuccessfully loaded and cleaned {len(df_clean)} rows.")
print("Proceeding to model training...")

# --- Preprocessing and Aggregation ---

# A. Feature Set 1: Unsorted (Original) Cell Voltages (Baseline)
X_unsorted = df_clean[CELL_FEATURES]
# The Linear Regression model performs the required aggregation (U1-U21 to Pack SOH).

# B. Feature Set 2: Sorted Cell Voltages (Comparison Preprocessing)
# Implements the 'sorting technique' preprocessing
X_sorted = np.sort(df_clean[CELL_FEATURES].values, axis=1)
X_sorted = pd.DataFrame(X_sorted, 
                        columns=[f'Sorted_U{i}' for i in range(1, 22)], 
                        index=df_clean.index) # Align index

# Target variable for all models
y = df_clean[TARGET]

# --- 2. Linear Regression Model Training & Evaluation ---

def train_and_evaluate_model(X, y, model_name):
    """Splits data, trains Linear Regression, and evaluates performance."""

    print(f"\n--- Training and Evaluation: {model_name} ---")

    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate Performance Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results = {'R²': r2, 'MSE': mse, 'MAE': mae, 'Model': model_name}

    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4e}")
    print(f"Mean Absolute Error (MAE): {mae:.4e}")

    return results, y_test.reset_index(drop=True), y_pred

# Train and collect results for both models
results_unsorted, y_test_unsorted, y_pred_unsorted = train_and_evaluate_model(
    X_unsorted, y, "Model 1: Unsorted Cell Voltages (Baseline)"
)
results_sorted, y_test_sorted, y_pred_sorted = train_and_evaluate_model(
    X_sorted, y, "Model 2: Sorted Cell Voltages (Preprocessed)"
)

# Comparison of Training Preprocessing Techniques
comparison_df = pd.DataFrame([results_unsorted, results_sorted])

print("\n--- Preprocessing Technique Comparison ---")
print(comparison_df.to_markdown(index=False, floatfmt=".4f"))

# --- 3. Threshold-Based Classification ---

def classify_battery_health(soh_prediction, threshold=0.6):
    """Implements the 0.6 threshold rule for battery classification."""
    if soh_prediction < threshold:
        return "Unhealthy (SOH < 0.6)"
    else:
        return "Healthy (SOH >= 0.6)"

# Example of classification using the predictions from Model 1
classification_threshold = 0.6 # This can be a variable entered by the user
sample_df = pd.DataFrame({
    'Actual SOH': y_test_unsorted.head(),
    'Predicted SOH': y_pred_unsorted[:5]
})
sample_df['Status'] = sample_df['Predicted SOH'].apply(lambda x: classify_battery_health(x, classification_threshold))

print(f"\n--- Classification Example using {classification_threshold} Threshold (Model 1) ---")
print(sample_df.to_markdown(index=False, floatfmt=".4f"))