import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from PIL import Image
import numpy as np
import pytesseract
import easyocr


# Load environment variables from .env
load_dotenv()

# Set up Gemini model using your API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{input}"),
])

# Output parser
output_parser = StrOutputParser()

# Combine the prompt, model, and parser into a chain
chain = prompt | llm | output_parser

# Streamlit UI
st.title('Language Translator using Gemini')

st.markdown(
    """
    <style>
    .st-emotion-cache-8atqhb.e1mlolmg0 {
        display: flex;}
    </style>
    """,
    unsafe_allow_html=True,
)

col_input, col_mic = st.columns([4, 1], vertical_alignment="bottom")

# Input box
with col_input:
    input_text = st.text_input("Enter text in any language:")

# Audio input
with col_mic:
    transcript = speech_to_text(key="stt", language="en", just_once=True, start_prompt="🎙️", stop_prompt="✔️")

if transcript:
    input_text = transcript
    st.write(f"Transcribed text: {input_text}")


# Image input

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    method = st.selectbox("OCR Engine", ["EasyOCR", "Pytesseract"])
    if st.button("Extract Text"):
        if method == "Pytesseract":
            text = pytesseract.image_to_string(img)
        else:
            reader = easyocr.Reader(['en'])
            text = "\n".join(reader.readtext(np.array(img), detail=0))
        input_text = text
        st.text_area("Extracted Text", text, height=250)

# Language selection
languages = [
    "Urdu", "German", "French", "Spanish", "Arabic", "Farsi",
    "Hindi", "Chinese", "Russian", "Turkish", "Japanese", "Italian"
]
selected_language = st.selectbox("Select language to translate to:", languages)

# Translation trigger
if input_text and selected_language:
    response = chain.invoke({
        "input_language":  selected_language,
        "output_language": selected_language,
        "input": input_text
    })

    st.markdown(f"### Translated ({selected_language}):")
    st.write(response)
