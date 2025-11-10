# src/core/data_handler.py
import pandas as pd
import streamlit as st
import io
from pypdf import PdfReader
import os
import tempfile
import polars as pl # NEW: Import polars

def load_data_from_upload(uploaded_file):
    """
    Loads data from an uploaded CSV or XLSX file into a polars DataFrame.
    """
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                # Use pl.read_csv for CSV
                df = pl.read_csv(io.BytesIO(uploaded_file.getvalue()))
            elif file_extension in ['xlsx', 'xls']:
                # Use pl.read_excel for XLSX
                # Ensure 'polars[excel]' is installed for this to work
                df = pl.read_excel(io.BytesIO(uploaded_file.getvalue()))
            else:
                st.error("Unsupported file type for DataFrame. Please upload a CSV or XLSX file.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def extract_text_from_document(file_input_object):
    """
    Extracts text content from a file-like object (e.g., Streamlit UploadedFile or standard binary file object).
    """
    text_content = ""
    file_bytes = None
    file_extension = ""
    # Determine if it's a Streamlit UploadedFile or a standard file object
    if hasattr(file_input_object, 'getvalue') and hasattr(file_input_object, 'name'):
        # It's a Streamlit UploadedFile
        file_bytes = file_input_object.getvalue()
        file_extension = file_input_object.name.split('.')[-1].lower()
    elif hasattr(file_input_object, 'read'):
        # It's a standard Python file object (e.g., from open(filepath, 'rb'))
        file_bytes = file_input_object.read()
        # Try to infer extension for standard file objects if name is not available
        if hasattr(file_input_object, 'name'): # If it's a file object opened with a name
            file_extension = file_input_object.name.split('.')[-1].lower()
        else: # Heuristic if only bytes are available (less reliable)
            if file_bytes.startswith(b'%PDF'):
                file_extension = 'pdf'
            else:
                file_extension = 'txt' # Default to txt
    else:
        st.warning("Unsupported file input type for text extraction. Must be UploadedFile or standard file object.")
        return ""
    if not file_bytes:
        st.warning("File object is empty or could not be read.")
        return ""
    try:
        if file_extension == 'txt':
            text_content = file_bytes.decode('utf-8')
        elif file_extension == 'pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text_content += page.extract_text() or "" # Handle pages with no extractable text
            # Limit PDF text to avoid excessive token usage for LLM, can be adjusted
            if len(text_content) > 10000: # Example limit
                st.warning("Document content truncated for processing due to length. Consider uploading smaller documents.")
                text_content = text_content[:10000] + "..."
        else:
            st.warning(f"Unsupported document type: .{file_extension}. Please upload a TXT or PDF file.")
            return ""
    except Exception as e:
        st.error(f"Error extracting text from document: {e}")
        return ""
    return text_content

def save_uploaded_file_to_temp(uploaded_file) -> str:
    """
    Saves an uploaded Streamlit file to a temporary file on disk.
    Returns the path to the temporary file.
    """
    try:
        # Create a temporary directory or file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file to temp: {e}")
        return None