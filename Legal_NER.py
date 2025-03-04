import streamlit as st
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import io

# Configure the API key
genai.configure(api_key = st.secrets["API_KEY"])

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
        - Date Filed
        - County
        - ACCT No
        - Property ID
        - Tax Amount
        - Defendant's Name
        - Defendant's Address
        Dont add any extra data in any field only add what is required (imp)
        Please analyze all pages of the document and provide the extracted information in a structured CSV format with correct headers also dont use , in Tax amount like if $6,385.56 write it as $6385.56. One more thing if there's multiple Defendants then write their data separately. Also in Defendant's Address if there's a , like 7 Clara Barton Ln Galveston,TX 77551 then replace it with 7 Clara Barton Ln Galveston;TX 77551. And don't use , in ACCT No values, write simply like 292600000002011155511 without commas in Raw csv data (imp)."""
    )
    input_prompt = [system_prompt, pdf_info[0]]
    response = model.generate_content(input_prompt)
    return response.text if response else ""

# Apply custom CSS for improved UI
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
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        @media (max-width: 600px) { /* Mobile */
            h1 {
                font-size: 20px;
            }
            .sidebar .sidebar-content {
                padding: 10px;
            }
        }
        @media (max-width: 1024px) and (min-width: 601px) { /* Tablet */
            h1 {
                font-size: 24px;
            }
        }
        @media (min-width: 1025px) { /* Laptop/PC */
            h1 {
                font-size: 30px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1>📜 Legal Document Information Extractor</h1>", unsafe_allow_html=True)

# Sidebar for file upload with styling
st.sidebar.markdown("""
    <div class="sidebar-content">
        <h2>📂 Upload PDF Files</h2>
        <p>Drag and drop PDF files to extract legal information.</p>
    </div>
""", unsafe_allow_html=True)

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
