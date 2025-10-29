"""
Combined AI Chatbot Application
Integrates Agent, Backend (FastAPI), and Frontend (Streamlit) into a single file
"""

import os
import sys
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ==================== AGENT SETUP ====================
# Step 1: Load API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step 2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as DeepSeekChat
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=2)

# Step 3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    """
    Get response from AI Agent based on LLM provider and model
    
    Args:
        llm_id: Model identifier
        query: User query/question
        allow_search: Boolean to enable/disable web search
        system_prompt: Custom system prompt for the AI
        provider: LLM provider (Groq or OpenAI)
    
    Returns:
        str: AI agent response
    """
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        if "deepseek" in llm_id.lower():
            llm = ChatOpenAI(
                model=llm_id,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools
    )
    state = {"messages": [{"role": "user", "content": query[0]}] if isinstance(query, list) else [{"role": "user", "content": query}]}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1] if ai_messages else "No response generated"


# ==================== BACKEND (FastAPI) ====================

class RequestState(BaseModel):
    """Request model for chat endpoint"""
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


from fastapi import FastAPI
import uvicorn

ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "deepseek/deepseek-chat-v3-0324"]

app = FastAPI(title="LangGraph AI Agent")


@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Create AI Agent and get response from it!
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response


# ==================== FRONTEND (Streamlit) ====================

def run_frontend():
    """Run Streamlit frontend"""
    import streamlit as st
    import requests

    st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
    st.title("THENABILMAN AI Chatbot Agents")
    st.write("Create and Interact with the AI Agents!")

    system_prompt = st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

    MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    MODEL_NAMES_OPENAI = ["deepseek/deepseek-chat-v3-0324"]

    provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

    if provider == "Groq":
        selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
    elif provider == "OpenAI":
        selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

    allow_web_search = st.checkbox("Allow Web Search")

    user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

    API_URL = os.getenv("API_URL", "http://127.0.0.1:9999/chat")

    if st.button("Ask Agent!"):
        if user_query.strip():
            # Step 2: Connect with backend via URL
            payload = {
                "model_name": selected_model,
                "model_provider": provider,
                "system_prompt": system_prompt,
                "messages": [user_query],
                "allow_search": allow_web_search
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=30)
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
                st.warning("Please make sure the backend is running. Run this command in another terminal:\n\n`python combined_app.py --backend`")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The AI agent is taking too long to respond.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    # Determine which component to run based on command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "frontend"
    
    if mode == "--backend":
        # Run backend API server
        print("Starting FastAPI Backend...")
        port = int(os.getenv("PORT", 9999))
        host = os.getenv("HOST", "127.0.0.1")
        
        # In production (Railway, Heroku, etc.), use 0.0.0.0
        if os.getenv("ENVIRONMENT") == "production":
            host = "0.0.0.0"
        
        uvicorn.run(app, host=host, port=port)
    else:
        # Run Streamlit frontend (default)
        print("Starting Streamlit Frontend...")
        run_frontend()
