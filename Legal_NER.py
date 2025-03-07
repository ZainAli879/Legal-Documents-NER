import streamlit as st
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import io

# Configure the API key
genai.configure(api_key="AIzaSyCw4clnJTbbLuarPmDlZmDI7FZglK7bBAY")

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

def gemini_output(pdf_path):
    """Extract structured data from the PDF using the Gemini model."""
    pdf_info = pdf_format(pdf_path)
    system_prompt = (
        """You are a specialist in extracting information from legal documents.
        Input PDFs in the form of legal documents will be provided to you,
        and your task is to extract and respond with the following information:
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
        You have to Extract First Name , Middle Name , Last Name of First Defandant only not all and Street No, Street Name , City Name , State Name , Zip Code of the same Defandant(imp) . also if only defandant's name is given but not location infomation then leave the fields Street No, Street Name , City Name , State Name , Zip Code empty and dot use other's defandants information
        If the document states "if living" AND "if any or all of the above-named Defendant(s) be deceased",
         → Extract "Deceased".
        If there is no mention of death after the Defendant’s details,
         → Leave the field empty.
        if there are multiple ACCT No or Property Id then you have to extract the first one only
        Dont provide multiple Defandants name and adresses only first Defandant and his/her address(imp)
        There will be only one record and no date files,county or anyother value should not be repeated
        In Property ID provide Property ID number not details of property
        Please analyze all pages of the document and provide the extracted information in a structured CSV format with correct headers also dont use , in Tax amount provide total aggregate Tax amount or Total Due of all properties and for example if total aggregate or Total Due is $6,385.56 write it as $6385.56 without using commas(,). And don't use , in ACCT No values, write simply like 292600000002011155511 without commas in Raw csv data (imp)."""
         )
    input_prompt = [system_prompt, pdf_info[0]]
    response = model.generate_content(input_prompt)
    return response.text if response else ""

# Apply custom CSS for dark & light mode compatibility
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
st.sidebar.header("Upload PDF Files")
st.sidebar.markdown("Drag and drop PDF files to extract legal information.")
uploaded_files = st.sidebar.file_uploader("Upload legal PDF documents", type=["pdf"], accept_multiple_files=True)

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
                # Clean up CSV formatting issues
                extracted_csv = extracted_csv.strip("`").replace("```csv", "").replace("```", "").strip()
                if extracted_csv.lower().startswith("csv"):
                    extracted_csv = extracted_csv[3:].strip()

                # Display raw extracted CSV data
                with st.expander("📑 Raw Extracted CSV Data"):
                    st.text_area("", extracted_csv, height=200)

                # Read CSV into a Pandas DataFrame
                df = pd.read_csv(io.StringIO(extracted_csv), header=0, engine='python', on_bad_lines="skip")

                # Display data in tabular format
                st.subheader("📊 Extracted Data in Tabular Format")
                st.dataframe(df, use_container_width=True)

                # Convert DataFrame to CSV for download
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
