import streamlit as st
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import io
import os

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

    **Important Extraction Rules:**
    - Extract **First Name, Middle Name, and Last Name** of **only the first Defendant**.
    - Extract **Street No, Street Name, City Name, State Name, and Zip Code** of the **same Defendant**.
    - If the document contains the phrases **"if living"** AND **"if any or all of the above-named Defendant(s) be deceased"**, then set **Deceased = "Deceased"**, otherwise leave it empty.
    - If multiple **Account No** or **Property ID** exist, extract **only the first one**.
    - **Do not repeat** any value (e.g., Case No, County, Date Filed, etc.).
    - In **Property ID**, provide the **Property ID number only**, not property details.
    - **Analyze all pages** to extract complete data.

    **CSV Formatting Rules:**
    - **No extra formatting**: Do **not** include triple backticks (` ``` `) or unnecessary symbols at the start or end of the extracted CSV data.
    - **Use a structured CSV format with proper headers**.
    - **No commas (`,`) in numerical values**:
      - **Tax Amount**: Provide the **total aggregate Tax Amount or Total Due** of all properties.  
        Example: If the Total Due is `$6,385.56`, write it as `$6385.56` (without commas).  
      - **Account No**: Write as a plain number without commas (e.g., `292600000002011155511`).
    
    Ensure the extracted output is **clean and formatted correctly** with **no extra characters or symbols**."""
  )

    input_prompt = [system_prompt, pdf_info[0]]
    response = model.generate_content(input_prompt)
    return response.text if response else ""

# Streamlit UI
st.markdown("<h1>📜 Legal Document Information Extractor</h1>", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload PDF Files")
st.sidebar.markdown("Drag and drop PDF files to extract legal information.")
uploaded_files = st.sidebar.file_uploader("Upload legal PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    combined_data = []  # List to store extracted data from all files

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
                extracted_csv = extracted_csv.strip().replace("csv", "").strip()
                if extracted_csv.lower().startswith("csv"):
                    extracted_csv = extracted_csv[3:].strip()

                # Convert CSV string to Pandas DataFrame
                df = pd.read_csv(io.StringIO(extracted_csv), header=0, engine='python', on_bad_lines="skip")

                # Display extracted data in a separate table
                st.subheader(f"📊 Data Extracted from {uploaded_file.name}")
                st.dataframe(df, use_container_width=True)

                # Append extracted data to the combined list
                combined_data.append(df)

            except Exception as e:
                st.error(f"⚠️ Error processing CSV for {uploaded_file.name}: {e}")
        else:
            st.error(f"❌ No relevant data found in {uploaded_file.name}. Please try another file.")

    # If data is extracted from at least one file, create a combined CSV (but don't display it)
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)  # Merge all DataFrames

        # Convert combined DataFrame to CSV for download
        combined_csv_file = combined_df.to_csv(index=False).encode("utf-8")

        # Show only one download button for all files
        st.download_button(
            label="📥 Download Combined Data",
            data=combined_csv_file,
            file_name="combined_extracted_data.csv",
            mime="text/csv",
            key="download_combined"
        )
    else:
        st.warning("⚠️ No valid data extracted from any file.")
