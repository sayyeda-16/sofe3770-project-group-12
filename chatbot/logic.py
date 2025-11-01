import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import traceback
import textwrap
import joblib

# load api key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# initialize client with the API key
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Why is the sky blue?")

# load the data set
def load_dataset(file_path="PulseBat Dataset.xlsx"):
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"Excel load failed: {e}. Using CSV")
        df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    return df

# load the pretrained models
def load_models():
    try:
        model_unsorted = joblib.load("models/model_unsorted.pkl")
        model_sorted = joblib.load("models/model_sorted.pkl")
        print("models loaded successfully")
        return model_unsorted, model_sorted
    except Exception as e:
        print(f"models failed to load: {e}")
        return None, None

def predict_soh(model, voltages):
    voltages = np.array(voltages).reshape(1, -1)
    prediction = model.predict(voltages)[0]
    return round(float(prediction), 4)

def classify_battery_health(soh_value, threshold=0.6):
    return "Healthy" if soh_value >= threshold else "Unhealthy"

def gemini_chat(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

def execute_dataset_query(df, code_str):
    try:
        allowed_names = {"df": df, "np": np}
        result = eval(code_str, {"__builtins__": {}}, allowed_names)
        return result
    except Exception:
        return f"error in executing query: \n{traceback.format_exc()}"

