# utils/resources.py
import streamlit as st
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def load_csv_with_encoding(file_path):
    """Load CSV file trying different encodings"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logging.info(f"Successfully loaded {file_path.name} with {encoding} encoding")
            
            # Rename 'No' column to 'ID' if it exists
            if 'No' in df.columns:
                df = df.rename(columns={'No': 'ID'})
            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"Error with {encoding}: {e}")
            continue
    
    # If all encodings fail, try without specifying encoding
    try:
        df = pd.read_csv(file_path, encoding=None, engine='python')
        logging.info(f"Loaded {file_path.name} with auto-detected encoding")
        
        if 'No' in df.columns:
            df = df.rename(columns={'No': 'ID'})
        
        return df
    except Exception as e:
        logging.error(f"Failed to load {file_path.name}: {e}")
        return pd.DataFrame()


def load_maternal_indicators():
    """Load maternal indicator descriptions from CSV file"""
    file_path = Path("utils/maternal_indicator_descripiton.csv")
    
    if file_path.exists():
        df = load_csv_with_encoding(file_path)
        if not df.empty:
            logging.info(f"Loaded {len(df)} maternal indicators")
        return df
    else:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()


def load_newborn_indicators():
    """Load newborn indicator descriptions from CSV file"""
    file_path = Path("utils/newborn_indicator_description.csv")
    
    if file_path.exists():
        df = load_csv_with_encoding(file_path)
        if not df.empty:
            logging.info(f"Loaded {len(df)} newborn indicators")
        return df
    else:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()


def render_resources_tab():
    """Render the resources tab with both maternal and newborn indicators"""
    
    st.markdown("""
    <h2 style="color: #2c3e50; margin-bottom: 20px; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
        📊 Indicator Reference Guide
    </h2>
    """, unsafe_allow_html=True)
    
    # Create tabs for maternal and newborn indicators
    tab1, tab2 = st.tabs(["Maternal Indicators", "Newborn Indicators"])
    
    with tab1:
        maternal_df = load_maternal_indicators()
        
        if not maternal_df.empty:
            st.markdown("#### Maternal Health Indicators")
            
            # Build HTML table
            html = """
            <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                margin: 15px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .custom-table thead {
                background-color: #2c3e50;
                color: white;
            }
            
            .custom-table th {
                padding: 14px 12px;
                text-align: left;
                font-weight: 600;
                border: none;
            }
            
            .custom-table td {
                padding: 12px 12px;
                border-bottom: 1px solid #e0e0e0;
                vertical-align: top;
                line-height: 1.5;
            }
            
            .custom-table tbody tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            
            .custom-table tbody tr:hover {
                background-color: #e8f4fd;
            }
            
            .custom-table tbody tr:last-child td {
                border-bottom: none;
            }
            
            /* Column widths */
            .col-id { width: 5%; text-align: center; }
            .col-name { width: 25%; }
            .col-numerator { width: 25%; text-align: justify; }
            .col-denominator { width: 25%; text-align: justify; }
            .col-formula { width: 20%; font-family: 'Courier New', monospace; }
            
            .text-justify {
                text-align: justify;
                text-justify: inter-word;
                hyphens: auto;
            }
            </style>
            
            <table class="custom-table">
            <thead>
                <tr>
                    <th class="col-id">ID</th>
                    <th class="col-name">Indicator Name</th>
                    <th class="col-numerator">Numerator</th>
                    <th class="col-denominator">Denominator</th>
                    <th class="col-formula">Formula</th>
                </tr>
            </thead>
            <tbody>
            """
            
            for _, row in maternal_df.iterrows():
                html += f"""
                <tr>
                    <td class="col-id">{row.get('ID', row.get('No', ''))}</td>
                    <td class="col-name"><strong>{row.get('Indicator Name', '')}</strong></td>
                    <td class="col-numerator text-justify">{row.get('Numerator', '')}</td>
                    <td class="col-denominator text-justify">{row.get('Denominator', '')}</td>
                    <td class="col-formula"><code>{row.get('Formula', '')}</code></td>
                </tr>
                """
            
            html += "</tbody></table>"
            
            # Use components.html to render HTML properly
            import streamlit.components.v1 as components
            components.html(html, height=600, scrolling=True)
            
            # FIXED DOWNLOAD: Create clean dataframe with plain text formulas
            download_df = maternal_df.copy()
            if 'Formula' in download_df.columns:
                download_df['Formula'] = download_df['Formula'].astype(str).str.replace('÷', ' divided by ').str.replace('×', ' multiplied by ').str.replace('Ã·', ' divided by ').str.replace('Ã—', ' multiplied by ')
            
            # Download button with fixed data
            st.download_button(
                "Download Maternal Indicators",
                data=download_df.to_csv(index=False),
                file_name="maternal_indicators.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.error("Failed to load maternal indicators.")
    
    with tab2:
        newborn_df = load_newborn_indicators()
        
        if not newborn_df.empty:
            st.markdown("#### Newborn Health Indicators")
            
            # Build HTML table
            html = """
            <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                margin: 15px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .custom-table thead {
                background-color: #2c5282;
                color: white;
            }
            
            .custom-table th {
                padding: 14px 12px;
                text-align: left;
                font-weight: 600;
                border: none;
            }
            
            .custom-table td {
                padding: 12px 12px;
                border-bottom: 1px solid #e0e0e0;
                vertical-align: top;
                line-height: 1.5;
            }
            
            .custom-table tbody tr:nth-child(even) {
                background-color: #f0f9ff;
            }
            
            .custom-table tbody tr:hover {
                background-color: #e0f2fe;
            }
            
            .custom-table tbody tr:last-child td {
                border-bottom: none;
            }
            
            /* Column widths */
            .col-id { width: 5%; text-align: center; }
            .col-name { width: 25%; }
            .col-numerator { width: 25%; text-align: justify; }
            .col-denominator { width: 25%; text-align: justify; }
            .col-formula { width: 20%; font-family: 'Courier New', monospace; }
            
            .text-justify {
                text-align: justify;
                text-justify: inter-word;
                hyphens: auto;
            }
            </style>
            
            <table class="custom-table">
            <thead>
                <tr>
                    <th class="col-id">ID</th>
                    <th class="col-name">Indicator Name</th>
                    <th class="col-numerator">Numerator</th>
                    <th class="col-denominator">Denominator</th>
                    <th class="col-formula">Formula</th>
                </tr>
            </thead>
            <tbody>
            """
            
            for _, row in newborn_df.iterrows():
                html += f"""
                <tr>
                    <td class="col-id">{row.get('ID', row.get('No', ''))}</td>
                    <td class="col-name"><strong>{row.get('Indicator Name', '')}</strong></td>
                    <td class="col-numerator text-justify">{row.get('Numerator', '')}</td>
                    <td class="col-denominator text-justify">{row.get('Denominator', '')}</td>
                    <td class="col-formula"><code>{row.get('Formula', '')}</code></td>
                </tr>
                """
            
            html += "</tbody></table>"
            
            # Use components.html to render HTML properly
            import streamlit.components.v1 as components
            components.html(html, height=500, scrolling=True)
            
            # FIXED DOWNLOAD: Create clean dataframe with plain text formulas
            download_df = newborn_df.copy()
            if 'Formula' in download_df.columns:
                download_df['Formula'] = download_df['Formula'].astype(str).str.replace('÷', ' divided by ').str.replace('×', ' multiplied by ').str.replace('Ã·', ' divided by ').str.replace('Ã—', ' multiplied by ')
            
            # Download button with fixed data
            st.download_button(
                "Download Newborn Indicators",
                data=download_df.to_csv(index=False),
                file_name="newborn_indicators.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.error("Failed to load newborn indicators.")