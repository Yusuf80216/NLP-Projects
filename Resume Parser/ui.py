import streamlit as st
import requests

def main():
    st.title("📄 Resume Parser")
    uploaded_resume = st.file_uploader(label="Upload your Resume", type=".pdf")
    button = st.button("Predict")
    
    if uploaded_resume is not None and button:
        with st.spinner('Processing...'):
            response = requests.post(url="http://127.0.0.1:8000/predict", files={'file': uploaded_resume})
            if response.status_code == 200:
                resume_category = response.json()['resume_category']
                st.write(f"**Resume Category:** {resume_category}")
            else:
                st.error("Failed to send resume to the API.")

if __name__ == "__main__":
    main()