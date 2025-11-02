import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
import joblib

# 1. API Client Initialization (Global & Efficient)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Print a warning instead of raising an error, allowing Task 1 functions to still work
    print("WARNING: GEMINI_API_KEY not found. Chatbot functionality will be disabled.")

# Initialize client and model globally for efficiency
try:
    genai.configure(api_key=api_key)
    # Define model globally for reuse in gemini_chat
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    GEMINI_MODEL = None  # Set to None if configuration failed


# 2. Data Loading and Cleaning (Task 1 Requirement)
def load_dataset(file_path="PulseBat Dataset.xlsx"):
    """
    Loads the dataset, cleans column headers, and removes rows with missing SOH or voltage data.
    """
    print(f"Loading data from {file_path}...")
    try:
        # Attempt to load Excel file (preferred)
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"Excel load failed: {e}. Attempting to load as CSV...")
        try:
            # Fallback to CSV
            df = pd.read_csv(file_path)
        except Exception as e_csv:
            raise FileNotFoundError(f"ERROR: Failed to load data as Excel or CSV: {e_csv}")

    # 1. Clean Column Headers
    df.columns = df.columns.str.strip()

    # 2. Remove rows with missing data (REQUIRED BY PROJECT README)
    voltage_cols = [f'U{i}' for i in range(1, 22)]
    required_cols = ['SOH'] + voltage_cols
    
    # Identify which required columns are actually present for robust cleaning
    cols_to_check = [col for col in required_cols if col in df.columns]
    
    initial_rows = len(df)
    df = df.dropna(subset=cols_to_check)
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing data in required columns.")
    
    print("Data loaded and cleaned successfully.")
    return df


# 3. Model Loading and Prediction (Task 1 Requirement)
def load_models():
    """Loads the pre-trained unsorted and sorted Linear Regression models."""
    try:
        model_unsorted = joblib.load("models/model_unsorted.pkl")
        model_sorted = joblib.load("models/model_sorted.pkl")
        print("Models loaded successfully.")
        return model_unsorted, model_sorted
    except FileNotFoundError:
        print("ERROR: Model files not found. Ensure models/model_unsorted.pkl and models/model_sorted.pkl exist.")
        return None, None
    except Exception as e:
        print(f"ERROR: Models failed to load: {e}")
        return None, None

def predict_soh(model, voltages):
    """
    Predicts SOH for a single sample using the given model.
    """
    REQUIRED_FEATURES = 21
    if len(voltages) != REQUIRED_FEATURES:
        raise ValueError(
            f"Input Error: Prediction requires exactly {REQUIRED_FEATURES} cell voltages (U1-U21), "
            f"but received {len(voltages)}. Please provide 21 values."
        )

    # Reshaping input for scikit-learn (1 sample, N features)
    voltages = np.array(voltages).reshape(1, -1)
    prediction = model.predict(voltages)[0]
    return round(float(prediction), 4)

def classify_battery_health(soh_value, threshold=0.6):
    """Classifies battery health based on the project's threshold (0.6)."""
    return "Healthy" if soh_value >= threshold else "Unhealthy"


# 4. Gemini Chat Functions (Task 2 Requirement)
def gemini_chat(prompt):
    """Generates content using the globally initialized GEMINI_MODEL."""
    if GEMINI_MODEL is None:
        return "Error: Gemini API key not configured or model initialization failed."
        
    try:
        # Use the globally configured model instance
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during the API call: {e}"

def execute_dataset_query(df, code_str, model_unsorted=None, predict_soh=None, classify_battery_health=None):
    """
    Executes a Python code string against the loaded DataFrame (df) and exposed functions.
    """
    try:
        # Allow access to the DataFrame, numpy, models, and functions
        allowed_names = {
            "df": df,
            "np": np,
            "model_unsorted": model_unsorted,
            "predict_soh": predict_soh,
            "classify_battery_health": classify_battery_health,
        }
        
        # Compile the code before evaluation (a minor security improvement)
        compiled_code = compile(code_str, "<string>", "eval")
        
        # The ValueError from predict_soh will be caught here
        result = eval(compiled_code, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        # Return a clear error message that app.py is designed to catch
        return f"Error in executing query. \nTraceback: {type(e).__name__}: {str(e).splitlines()[-1]}"
