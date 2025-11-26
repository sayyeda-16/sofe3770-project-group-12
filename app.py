import streamlit as st
import re # We need this for robust code block parsing
from chatbot.logic import (
    load_dataset,
    load_models,
    predict_soh,
    classify_battery_health,
    gemini_chat,
    execute_dataset_query,
)

# 1. Initialization and Caching (Crucial for Performance)

# Use st.cache_data to run this function only once.
# This prevents the large dataset file and models from being reloaded 
# on every single user interaction.
@st.cache_data(show_spinner="Loading dataset and models...")
def initialize_data_and_models():
    """Load the dataset and models once using Streamlit caching."""
    # Note: load_dataset should not take an argument here, but if your load_dataset
    # is modified to use a default, it's safer to pass the file path if it's external.
    data_df = load_dataset("PulseBat Dataset.xlsx")
    unsorted_model, sorted_model = load_models()
    return data_df, unsorted_model, sorted_model

# Load data and models globally once
df, model_unsorted, model_sorted = initialize_data_and_models()

# Initialize chat history using standard 'messages' key
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Battery SOH Assistant. Ask me a general question, or query the dataset (e.g., 'What is the average SOH?')"}
    ]


# 2. Robust Gemini Prompt (Forces Code Block Output)

SYSTEM_PROMPT = f"""
You are an expert assistant for a battery SOH project.
The primary dataset is stored in a DataFrame named 'df' with the following columns: {', '.join(df.columns)}.
You have access to the following Python functions: 
1. `predict_soh(model_unsorted, voltages_list)`: Used for SOH prediction. Requires **exactly 21 voltage values** in the list.
2. `classify_battery_health(soh_value)`: Used to classify the prediction as 'Healthy' or 'Unhealthy'.

**CRITICAL INSTRUCTIONS:**
- If the user asks a general conceptual question, answer directly.
- **If the user asks for data statistics or a prediction, you MUST respond with a single, self-contained, executable Python code block.**
- **The code block MUST start with ```python and end with ```**
- **The code block MUST contain ONLY a single Python expression that returns a final result.** - **DO NOT use variable assignments (e.g., DO NOT use `voltages = [...]`).**

- **FIX/EXAMPLE for Prediction:** If the user provides 21 voltage values, generate the code like this (using the actual 21 values provided by the user):
  ```python
  classify_battery_health(predict_soh(model_unsorted, [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21])"""


# SIDEBAR
st.sidebar.title("🔋 Battery Analytics")
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"**Samples:** {len(df)}")
st.sidebar.write(f"**Features:** {len(df.columns)}")

if 'SOH' in df.columns:
    st.sidebar.write(f"**Avg SOH:** {df['SOH'].mean():.2f}")
    st.sidebar.write(f"**SOH Range:** {df['SOH'].min():.2f} - {df['SOH'].max():.2f}")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# 3. Streamlit Application and Execution Logic

st.set_page_config(page_title="Battery SOH Chatbot", layout="wide")
st.title("Battery SOH Chatbot: Dataset Exploration and SOH Prediction Assistant")

if st.button("🧹 Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Battery SOH Assistant. How can I help today?"}
    ]
    st.rerun()
# Display Chat History
for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Handle User Input
if prompt := st.chat_input("Ask about SOH predictions or dataset insights..."):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get LLM response using the robust system prompt
    with st.spinner("Thinking..."):
        llm_output = gemini_chat(SYSTEM_PROMPT + "\n\nUser Question: " + prompt)

    # 3. Look for a Python code block using regex
    code_match = re.search(r"```python\n(.*?)\n```", llm_output, re.DOTALL)
    
    if code_match:
        # A code block was found, execute it.
        code_str = code_match.group(1).strip()
        
        # Execute the code against the cached dataframe/models
        # We pass model_unsorted and classify_battery_health for execution
        result = execute_dataset_query(
            df=df,
            code_str=code_str,
            model_unsorted=model_unsorted,
            predict_soh=predict_soh,
            classify_battery_health=classify_battery_health,
        )

        # 4. Handle Execution Result
        if isinstance(result, str) and result.startswith("Error in executing query"):
            # Execution failed, show error message to the user
            final_response = f"**Execution Error:** Your query failed to run. Please check the function syntax or the DataFrame column names. \n\n*Error details:* {result}"
        else:
            # Execution succeeded, send result back to Gemini for interpretation
            feedback_prompt = (
                f"You executed the code: `{code_str}`. The final result was: `{result}`. "
                "Explain this final result to the user in a friendly, conversational way."
            )
            final_response = gemini_chat(feedback_prompt)

        # Display code executed (for transparency) and final response
        with st.chat_message("assistant"):
            st.markdown(f"**Code Executed:** `{code_str}`")
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
    else:
        # No code block was found, treat as a general conceptual question
        with st.chat_message("assistant"):
            st.markdown(llm_output)
            st.session_state.messages.append({"role": "assistant", "content": llm_output})
