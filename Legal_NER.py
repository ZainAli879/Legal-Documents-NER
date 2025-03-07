import streamlit as st
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import io

# Access the API key from Streamlit secrets
api_key = st.secrets["API_KEY"]

# Configure the API key for genai
genai.configure(api_key=api_key)

# Model Configuration
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Load the model
model = genai.GenerativeModel(
    model_name="models/gemini-2.0-pro-exp-02-05",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

def pdf_format(pdf_path):
    """Prepare the PDF file for the Gemini model."""
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"Could not find PDF: {pdf}")
    return [{"mime_type": "application/pdf", "data": pdf.read_bytes()}]

def process_text_input(text_data):
    """Process text input and extract structured data."""
    system_prompt = (
        """You are a specialist in extracting structured legal data.
        The text data you receive contains legal case information. 
        Your task is to extract and respond with the following fields in CSV format:
        - Case No
        - County
        - Date Filed
        - First Name
        - Middle Name
        - Last Name
        - Street No
        - Street Name
        - City Name
        - State Name
        - Zip Code
        - Deceased
        - Account No
        - Property ID
        - Tax Amount

        Rules:
        - Only extract the **first defendant's** name and address.
        - If "if living" AND "if any or all of the above-named Defendant(s) be deceased" exist, mark "Deceased".
        - If multiple Account No or Property IDs exist, extract only the **first** one.
        - Ensure Tax Amount has no commas (e.g., $6385.56 not $6,385.56).
        - Format CSV **without** redundant headers or repeated values.
        """
    )
    
    input_prompt = [system_prompt, text_data]
    response = model.generate_content(input_prompt)
    return response.text if response else ""

def gemini_output(pdf_path):
    """Extract structured data from the PDF using the Gemini model."""
    pdf_info = pdf_format(pdf_path)
    system_prompt = (
        """You are a specialist in extracting information from legal documents.
        Extract the following fields in CSV format, following the same rules as the text-based function.
        """
    )
    input_prompt = [system_prompt, pdf_info[0]]
    response = model.generate_content(input_prompt)
    return response.text if response else ""

# Apply custom CSS for styling
st.markdown("""
    <style>
        html, body, [class*="stApp"]  {
            font-family: 'Arial', sans-serif;
        }
        h1 {
            text-align: center;
        }
        .stDownloadButton button {
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            transition: background-color 0.3s ease;
        }
        @media (prefers-color-scheme: dark) {
            h1, h2, h3, h4, h5, h6, p {
                color: white;
            }
            .stDownloadButton button {
                background-color: #1f77b4;
                color: white;
            }
        }
        @media (prefers-color-scheme: light) {
            h1, h2, h3, h4, h5, h6, p {
                color: black;
            }
            .stDownloadButton button {
                background-color: #0a74da;
                color: white;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1>📜 Legal Document Information Extractor</h1>", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload PDF or Enter Text")
uploaded_files = st.sidebar.file_uploader("Upload legal PDF documents", type=["pdf"], accept_multiple_files=True)

# Text input option
text_data = st.sidebar.text_area("Or paste legal case details below:")

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.subheader(f"📄 Extracted Information: {uploaded_file.name}")

        # Extract CSV data from the Gemini model
        extracted_csv = gemini_output(pdf_path)

        if extracted_csv.strip():
            try:
                extracted_csv = extracted_csv.strip("`").replace("```csv", "").replace("```", "").strip()
                if extracted_csv.lower().startswith("csv"):
                    extracted_csv = extracted_csv[3:].strip()

                with st.expander("📑 Raw Extracted CSV Data"):
                    st.text_area("", extracted_csv, height=200)

                df = pd.read_csv(io.StringIO(extracted_csv), header=0, engine='python', on_bad_lines="skip")

                st.subheader("📊 Extracted Data in Tabular Format")
                st.dataframe(df, use_container_width=True)

                csv_file = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Data",
                    data=csv_file,
                    file_name=f"extracted_data_{uploaded_file.name}.csv",
                    mime="text/csv",
                    key=f"download_{uploaded_file.name}"
                )
            except Exception as e:
                st.error(f"⚠️ Error processing CSV: {e}")
        else:
            st.error(f"❌ No relevant data found in {uploaded_file.name}. Please try another file.")

elif text_data:
    st.subheader("📄 Extracted Information from Text")

    extracted_csv = process_text_input(text_data)

    if extracted_csv.strip():
        try:
            extracted_csv = extracted_csv.strip("`").replace("```csv", "").replace("```", "").strip()
            if extracted_csv.lower().startswith("csv"):
                extracted_csv = extracted_csv[3:].strip()

            with st.expander("📑 Raw Extracted CSV Data"):
                st.text_area("", extracted_csv, height=200)

            df = pd.read_csv(io.StringIO(extracted_csv), header=0, engine='python', on_bad_lines="skip")

            st.subheader("📊 Extracted Data in Tabular Format")
            st.dataframe(df, use_container_width=True)

            csv_file = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Data",
                data=csv_file,
                file_name="extracted_text_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Error processing CSV: {e}")
    else:
        st.error("❌ No relevant data found. Please check your input.")

