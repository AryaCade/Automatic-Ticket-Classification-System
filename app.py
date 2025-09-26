# app.py 
import os
import getpass
import re
import json
import pandas as pd
from datetime import datetime
from os.path import exists

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please add it in your `.env` file.")
    st.stop()  # stop execution if no key


# Initialize the LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


system_prompt = "You are a intellegent assistent designed to help human agents route incoming cases to the apropreate team. Your roal is to analyse case descriptions, identify key information and recommend the most sutable team based on the provided guidlines."


classification_prompt = """# Automatic Ticket Classification\n

            You are provided with a description of a customer support case. Your task is to classify the case into one of the following categories based on the content of the description:\n

            1. ## Queue Name: Billing Issue
               **Context:**
               Cases related to invoicing, payments, refunds, or any financial transactions.\n

               **Example:**
                - "I was charged twice for my last purchase. Please help me get a refund."
                - "Can you explain the charges on my latest invoice?"
                - "I need to update my payment method for future subscriptions."\n

            2. ## Queue Name: Technical Support
               **Context:**
               Issues related to product functionality, bugs, errors, or technical difficulties.\n

                **Example:**
                - "The app crashes every time I try to open it."
                - "I'm experiencing connectivity issues with my device."
                - "How do I reset my password?"\n

            3. ## Queue Name: Account Management
               **Context:**
               Requests for account updates, password resets, profile changes, or account deletions.\n

                **Example:**
                - "I need to change the email address associated with my account."
                - "How do I delete my account?"
                - "I forgot my password and can't log in."\n

            4. ## Queue Name: General Inquiry
               **Context:**
               Any other questions or requests that do not fit into the above categories.\n

                **Example:**
                - "What are your business hours?"
                - "Where can I find more information about your services?"
                - "Can you provide details about your return policy?"\n

            5. ## Queue Name: Feedback/Suggestions
               **Context:**
               Customer feedback, suggestions for improvements, or feature requests.\n

                **Example:**
                - "I love your product! It would be great if you could add a dark mode feature."
                - "The user interface could be more intuitive."
                - "I have some suggestions to improve your service."\n

            6. ## Queue Name: Cancellation/Termination
               **Context:**
               Requests to cancel services, subscriptions, or terminate accounts.\n

                **Example:**
                - "I would like to cancel my subscription effective immediately."
                - "Please terminate my account and delete all associated data."
                - "How do I stop my recurring payments?"\n

            7. ## Queue Name: Website issue
                **Context:**
                Problems related to website functionality, navigation, or accessibility.\n

                **Example:**
                - "The checkout page is not loading properly."
                - "I can't find the product I am looking for on your website."
                - "The website is not displaying correctly on my mobile device."\n

            8. ## Queue Name: App issue
                **Context:**
                Issues related to mobile or desktop applications, including crashes, bugs, or performance problems.\n

                **Example:**
                - "The app crashes when I try to upload a photo."
                - "I'm experiencing slow performance on the latest version of your app."
                - "How do I update the app to the newest version?"\n

            Please read the case description carefully and provide your classification along with a brief explanation of your reasoning.\n

            ⚠️ IMPORTANT: Your answer must be in **valid JSON format** with exactly two fields:
            - "queue" → the chosen category
            - "reason" → a short explanation

            ### Case Description:
            {case_description}

            ### Your Classification (JSON only):"""





# ================== Streamlit UI ==================
st.set_page_config(page_title="SmileBird Support Bot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    h1 { color: #ffffff; text-align: center; }
    .stButton>button { background-color: #1db954; color: white; border: none; border-radius: 12px; font-size: 16px; font-weight: bold; padding: 0.6em 1.2em; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #17a34a; transform: scale(1.05); }
    .stTextArea textarea { background-color: #1e1e1e; color: #ffffff; border-radius: 8px; }
    .stDataFrame { background-color: #1e1e1e !important; color: #e0e0e0 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 SmileBird Support Bot")
st.write("Type your issue below and I'll classify it!")


# ================== Prediction ==================
def predict_with_llm(text: str):
    human_prompt = classification_prompt.format(case_description=text)
    messages = [("system", system_prompt), ("human", human_prompt)]

    ai_msg = llm.invoke(messages)

    raw_content = ai_msg.content.strip()

    # Clean common formatting issues
    raw_content = raw_content.strip("` \n")  # remove backticks/newlines
    if raw_content.lower().startswith("json"):
        raw_content = raw_content[4:].strip()  # remove leading 'json'

    # Extract JSON object with regex if extra text exists
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        raw_content = match.group(0)

    try:
        response = json.loads(raw_content)
        # Always enforce expected keys
        return {
            "queue": response.get("queue", "General Inquiry"),
            "reason": response.get("reason", "No reason provided"),
        }
    except Exception:
        return {"queue": "General Inquiry", "reason": raw_content}



if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

user_input = st.text_area("Enter your problem:")

if st.button("Classify") and user_input.strip():
    result = predict_with_llm(user_input)
    st.session_state.prediction_result = {
        "issue": user_input,
        "category": result["queue"],
        "reason": result["reason"],
    }


# ================== Display Results ==================
if st.session_state.prediction_result:
    result = st.session_state.prediction_result

    st.success(f"**Category:** {result['category']}")
    st.write("### Reason:")
    st.write(result["reason"])

    # Save Ticket
    if st.button("Save Ticket"):
        log_entry = pd.DataFrame([[
            datetime.now(),
            result["issue"],
            result["category"],   # ✅ already has queue
            result["reason"]
        ]], columns=["Timestamp", "Issue", "Team", "Reason"])


        file_exists = exists("tickets_log.csv")
        log_entry.to_csv("tickets_log.csv", mode="a", header=not file_exists, index=False)

        st.success("Ticket saved successfully!")
        st.session_state.prediction_result = None
        st.rerun()


# ================== History Tab ==================
if st.checkbox("📜 Show Ticket History"):
    try:
        df = pd.read_csv("tickets_log.csv")
        st.dataframe(df.iloc[::-1], use_container_width=True)
    except FileNotFoundError:
        st.info("No tickets logged yet.")
    except pd.errors.EmptyDataError:
        st.info("The ticket log is empty.")