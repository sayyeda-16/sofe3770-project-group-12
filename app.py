# app.py

import streamlit as st
from chatbot.logic import (
    load_dataset,
    load_models,
    predict_soh,
    classify_battery_health,
    gemini_chat,
    execute_dataset_query,
)

# ---- Load Dataset and Models ----
df = load_dataset("PulseBat Dataset.xlsx")
model_unsorted, model_sorted = load_models()

# ---- Streamlit App ----
st.set_page_config(page_title="Battery SOH Chatbot", layout="wide")
st.title("🔋 Battery SOH Chatbot")
st.caption("Ask questions about the dataset, get SOH predictions, or learn about batteries!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask about SOH predictions or dataset insights...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    system_prompt = f"""
    You are an assistant for a battery dataset.
    The dataset includes columns: {', '.join(df.columns)}.

    - If asked about data stats (mean, median, max, etc.), output valid Python using 'df'.
      Example: df['SOH'].mean()
    - If asked to predict SOH, respond with: predict_soh([...])
    - Otherwise, explain conceptually.
    """

    llm_output = gemini_chat(system_prompt + "\n\nUser: " + user_input)

    if "df" in llm_output or "predict_soh" in llm_output:
        try:
            if "predict_soh" in llm_output:
                voltages = [float(x) for x in llm_output.split("[")[1].split("]")[0].split(",")]
                prediction = predict_soh(model_unsorted, voltages)
                result = f"Predicted SOH: **{prediction}** ({classify_battery_health(prediction)})"
            else:
                result = execute_dataset_query(df, llm_output)
        except Exception as e:
            result = f"⚠️ Execution error: {e}"
    else:
        result = llm_output

    st.session_state.history.append({"role": "assistant", "content": str(result)})

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
