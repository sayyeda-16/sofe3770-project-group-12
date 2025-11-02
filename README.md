# SOFE3370 Project Task 1: Linear Regression for Battery SOH Prediction

This submission covers the first project milestone: training and evaluating a **Linear Regression** model for predicting the **State of Health (SOH)** of a battery pack using the **PulseBat dataset**.

## \**These are the following steps to run the updated code, which includes the Gemini chatbot that answers questions about the dataset or general questions about batteries.**
1) If you don't have the project, clone the repo; otherwise, pull the latest changes using `git pull origin main`
2) Install dependencies listed in the requirements.txt file using `pip install -r requirements.txt`
3) Search gemini api key, get an api key, create a .env file, and put `GEMINI_API_KEY="yourapikeyhere"` (make sure not to commit the env file to git)
4) Run the model: `python linear_regression_soh_prediction.py`
5) Run the Streamlit app to chat with the Gemini model: `streamlit run app.py`

## 1. Instructions on How to Set Up and Run the Code

### 1.1 Dependencies

Ensure you have **Python installed** (version 3.8+ is recommended). The following libraries are required:

| Library | Purpose | Installation Command |
| :--- | :--- | :--- |
| **pandas** | Data loading and manipulation | `pip install pandas` |
| **numpy** | Numerical operations (e.g., sorting) | `pip install numpy` |
| **scikit-learn** | Machine Learning models and metrics | `pip install scikit-learn` |
| **openpyxl** | Required for reading .xlsx files | `pip install openpyxl` |
| **tabulate** | Required for printing Markdown tables | `pip install tabulate` |

You can install all necessary dependencies at once using the following command: `pip install pandas numpy scikit-learn openpyxl tabulate`

### 1.2 File Structure

The following files must be in the same directory:

- `linear_regression_soh_prediction.py` (The Python script)  
- `PulseBat Dataset.xlsx` (The dataset file)  
- `README.md` (This document)  

### 1.3 Running the Code

Execute the Python script from the terminal or command prompt: `python linear_regression_soh_prediction.py`

### 1.4 Output

The script will output the following sections, demonstrating the completion of the task:

- Status messages confirming successful data loading
- Evaluation metrics (\( R^2 \), MSE, MAE) for the Unsorted model
- Evaluation metrics (\( R^2 \), MSE, MAE) for the Sorted model
- A comparison table of the preprocessing techniques
- A demonstration of the threshold-based classification

## 2. Technical Implementation Details

### 2.1 Dataset Handling and Preprocessing

- **File Handling**: The code uses robust exception handling to read the `PulseBat Dataset.xlsx` file. It first attempts to use `pd.read_excel` (requiring `openpyxl`) and falls back to `pd.read_csv`, ensuring compatibility with the file format used in the project
- **Data Cleaning**: Column headers are cleaned using `.str.strip()` to remove unseen whitespace, which prevents `KeyError` exceptions. Rows with missing data (`NaN`) in the SOH or cell voltage columns are removed
- **Aggregation (U1–U21 to Pack SOH)**: The Linear Regression model inherently performs the aggregation. It learns optimal coefficients for all 21 cell voltages (\( U1 \dots U21 \)) to combine them into a single, estimated Pack SOH value

### 2.2 Preprocessing Technique Comparison

Two distinct feature sets were created and compared to assess the impact of preprocessing:

| Model              | Feature Set             | Implementation                                                                 | Comparison Result                                                                 |
|--------------------|-------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Model 1 (Baseline) | Unsorted Cell Voltages  | Uses the physical voltage readings in their original order (\( U1, U2, ..., U21 \)) | Provides context and location-specific data to the model                          |
| Model 2 (Comparison)| Sorted Cell Voltages   | For every data row, the 21 voltages are sorted numerically, creating new features \[(U_min, ..., U_max)] | Demonstrated Marginal Superiority (\( R^2 = 0.6588 \) vs \( 0.6561 \)), suggesting that normalizing cell variation slightly improves prediction accuracy |


### 2.3 Linear Regression Model & Evaluation

- **Model**: `sklearn.linear_model.LinearRegression` was used
- **Data Split**: 80% Training / 20% Testing (`random_state=42`)
- **Metrics**: The model's predictive performance was evaluated using the required metrics:
  - R² (goodness of fit)
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)

### 2.4 Threshold-based Classification

The logic for determining the battery's health status was implemented in the `classify_battery_health` function.

**Rule**: The prediction from the Linear Regression model is used with the variable threshold:

- If SOH \( >= 0.6 \): Status is **Healthy**
- If SOH \( < 0.6 \): Status is **Unhealthy**

