import streamlit as st
import requests

st.set_page_config(
    page_title="Text Generation Studio",
    page_icon="🤖",
    layout="wide"
)

# Sidebar - Settings & Backend Config
st.sidebar.title("⚙️ Generation Settings")
backend_url = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

# Check backend health
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        if health_data.get("model_loaded"):
            st.sidebar.success(f"Backend Online ({health_data.get('device', 'cpu').upper()})")
        else:
            st.sidebar.warning("Backend Online, but model is not loaded yet.")
    else:
        st.sidebar.error("Backend returned an error.")
except Exception:
    st.sidebar.error("Backend Offline. Start FastAPI on port 8000.")

st.sidebar.markdown("---")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=256, value=50, step=5)
temperature = st.sidebar.slider("Temperature", min_value=0.05, max_value=1.5, value=0.7, step=0.05)
top_p = st.sidebar.slider("Top-p (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05)

# Main UI
st.title("🤖 Text Generation Playground")
st.markdown("Interact directly with the fine-tuned text generation model served via FastAPI.")

prompt = st.text_area(
    "Enter your prompt (optional):",
    value="",
    placeholder="Type a prompt to continue from, or leave empty for unconditional generation...",
    height=140
)

col1, col2 = st.columns([1, 5])
with col1:
    generate_btn = st.button("Generate Text", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner("Generating..."):
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 50,
                "repetition_penalty": repetition_penalty
            }
            response = requests.post(f"{backend_url}/generate", json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("generated_text", "")
                tokens = data.get("tokens_generated", 0)
                latency = data.get("latency_seconds", 0.0)
                speed = round(tokens / latency, 2) if latency > 0 else 0

                st.markdown("### Output")
                st.text_area("Full Output", value=f"{prompt}{generated_text}", height=200, disabled=True)

                # Performance Metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Latency", f"{latency:.2f} s")
                metric_col2.metric("Tokens Generated", f"{tokens}")
                metric_col3.metric("Throughput", f"{speed} tok/s")
            else:
                error_detail = response.json().get("detail", response.text)
                st.error(f"Generation error ({response.status_code}): {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
