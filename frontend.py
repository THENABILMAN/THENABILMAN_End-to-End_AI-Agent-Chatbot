
import streamlit as st
import os

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("THENABILMAN AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["deepseek/deepseek-chat-v3-0324"]

provider=st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search=st.checkbox("Allow Web Search")

user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL=os.getenv("API_URL", "http://127.0.0.1:9999/chat")

if st.button("Ask Agent!"):
    if user_query.strip():
        #Step2: Connect with backend via URL
        import requests

        payload={
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response=requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    st.subheader("Agent Response")
                    st.markdown(f"**Final Response:** {response_data}")
            else:
                st.error(f"Error: Server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the backend server!")
            st.warning("Please make sure the backend is running. Run this command in another terminal:\n\n`python backend.py`")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The AI agent is taking too long to respond.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


