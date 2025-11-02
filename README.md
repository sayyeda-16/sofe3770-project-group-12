# SOFE3370 Project Documentation

This repository contains the code and documentation for the SOFE3370 project, structured into two key tasks: **Task 1: Battery State of Health (SOH) Prediction** using Linear Regression, and **Task 2: Dataset Exploration** using the Gemini API and Streamlit.

---

## Getting Started

This section details the steps required to set up the environment for both tasks.

### 1. Project Initialization

1.  **Clone the Repo or Pull Changes:**
    * If you don't have the project: `git clone <your_repo_link>`
    * Otherwise, pull the latest changes: `git pull origin main`

2.  **Install Dependencies:**
    * Install all Python packages required for the entire project. The required libraries are listed below, and all can be installed at once using the `requirements.txt` file.
        ```bash
        pip install -r requirements.txt
        ```

| Library | Purpose |
| :--- | :--- |
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical operations (e.g., sorting) |
| **scikit-learn** | Machine Learning models and metrics |
| **openpyxl** | Required for reading `.xlsx` files |
| **tabulate** | Required for printing Markdown tables |
| **streamlit** | Web application framework for the chat UI |
| **google-genai** | Python library for interacting with the Gemini API |

### 2. API Key & Chatbot Setup (Required for Task 2)

**This step is mandatory to run the interactive chat application (`app.py`).**

1.  **Search & Get Your Gemini API Key:**
    * You must first obtain a key from the Google AI developer platform.
    * **Action:** Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and generate a new API key. **Copy this key immediately.**

2.  **Create and Populate the `.env` File:**
    * **Action:** In the project's root directory, create a file named **`.env`**.
    * **Action:** Open the file and set the environment variable, replacing `your_actual_api_key_here` with the key you just copied:
        ```
        GEMINI_API_KEY="your_actual_api_key_here"
        ```

3.  **Secure Your Key (CRITICAL):**
    * **Action:** Open your **`.gitignore`** file and ensure the following line is present to prevent your key from being committed to Git:
        ```
        .env
        ```

---

## Task 1: Linear Regression for Battery SOH Prediction

This section covers training and evaluating a **Linear Regression** model for predicting the **State of Health (SOH)** of a battery pack using the **PulseBat dataset**.

### 1.1 Running the SOH Model

* **File:** `linear_regression_soh_prediction.py`
* **Prerequisite:** Ensure the `PulseBat Dataset.xlsx` file is in the root directory.
* **Execution:** Run the Python script from your terminal:
    ```bash
    python linear_regression_soh_prediction.py
    ```
* **Output:** The script will output the following sections:
    * Status messages confirming successful data loading.
    * Evaluation metrics ($\mathbf{R^2}$, MSE, MAE) for both the Unsorted and Sorted models.
    * A comparison table of the preprocessing techniques.
    * A demonstration of the threshold-based classification.

### 1.2 Technical Implementation Details

#### Dataset Handling and Preprocessing
* **File Handling**: The code uses robust exception handling to read the `PulseBat Dataset.xlsx` file.
* **Data Cleaning**: Column headers are cleaned using `.str.strip()` to remove unseen whitespace. Rows with missing data (`NaN`) in the SOH or cell voltage columns are removed.
* **Aggregation ($\mathbf{U1–U21}$ to Pack SOH)**: The Linear Regression model inherently performs the aggregation. It learns optimal coefficients for all 21 cell voltages ($\mathbf{U1 \dots U21}$) to combine them into a single, estimated Pack SOH value.

#### Preprocessing Technique Comparison
Two distinct feature sets were created and compared to assess the impact of preprocessing:

| Model | Feature Set | Implementation | Key Finding |
| :--- | :--- | :--- | :--- |
| **Model 1 (Baseline)** | Unsorted Cell Voltages | Uses the physical voltage readings in their original order ($\mathbf{U1, U2, ..., U21}$) | Provides context and location-specific data to the model. |
| **Model 2 (Comparison)** | Sorted Cell Voltages | For every data row, the 21 voltages are sorted numerically, creating new features $\mathbf{[U_{min}, ..., U_{max}]}$ | Demonstrated **Marginal Superiority** ($\mathbf{R^2 = 0.6588}$ vs $\mathbf{0.6561}$), suggesting that normalizing cell variation slightly improves prediction accuracy. |

#### Model & Evaluation
* **Model**: `sklearn.linear_model.LinearRegression` was used.
* **Data Split**: **80% Training / 20% Testing** ($\mathbf{random\_state=42}$).
* **Metrics**: The model's predictive performance was evaluated using the required metrics: $\mathbf{R^2}$ (goodness of fit), **MSE** (Mean Squared Error), and **MAE** (Mean Absolute Error).
* **Threshold-based Classification**: The logic for determining the battery's health status was implemented using a variable threshold:
    * If $\mathbf{SOH \ge 0.6}$: Status is **Healthy**
    * If $\mathbf{SOH < 0.6}$: Status is **Unhealthy**

---

## Task 2: Streamlit Chatbot for Dataset Exploration

This section details how to run the interactive Streamlit application that uses the Gemini model to answer battery-related and dataset-specific questions.

### 2.1 Running the Chatbot App

* **File:** `app.py`
* **Prerequisite:** Ensure you have successfully completed the **API Key & Chatbot Setup** (Section 2).
* **Execution:** Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
* **Usage:** This command will launch the interactive chat application in your default web browser, allowing you to ask questions about the dataset or general battery information.
