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


# Add caching to the load functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
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


@st.cache_data(ttl=3600)  # Cache for 1 hour
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
    """Render the resources tab with lazy loading"""
    
    # Initialize session state for lazy loading
    if 'resources_loaded' not in st.session_state:
        st.session_state.resources_loaded = False
    if 'active_resource_tab' not in st.session_state:
        st.session_state.active_resource_tab = "maternal"
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 25px; color: white;">
        <h1 style="margin: 0; font-size: 2rem;">📊 Indicator Reference Guide</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">
            Comprehensive reference for maternal and newborn health indicators
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if resources should be loaded
    if not st.session_state.resources_loaded:
        # Show a simple load button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 1rem; background: #f8f9fa;
                     border-radius: 12px; border: 2px dashed #dee2e6; margin: 2rem 0;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
                    <h2 style="color: #495057; margin-bottom: 1rem;">Resources Dashboard</h2>
                    <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                        Click below to load indicator definitions and formulas
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            if st.button(
                "Load Resources",
                use_container_width=True,
                type="primary",
                key="load_resources_btn"
            ):
                st.session_state.resources_loaded = True
                st.rerun()
        return
    
    # Create tabs for maternal and newborn indicators
    tab1, tab2 = st.tabs(["🤰 **Maternal Indicators**", "👶 **Newborn Indicators**"])
    
    with tab1:
        # Update active tab state
        if st.session_state.active_resource_tab != "maternal":
            st.session_state.active_resource_tab = "maternal"
        
        # Load maternal indicators (cached)
        with st.spinner("Loading maternal indicators..."):
            maternal_df = load_maternal_indicators()
        
        if not maternal_df.empty:
            st.markdown("#### 🤰 Maternal Health Indicators")
            
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
                background: linear-gradient(135deg, #2e7d32, #1b5e20);
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
                background-color: #e8f5e9;
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
                "📥 Download Maternal Indicators",
                data=download_df.to_csv(index=False),
                file_name="maternal_indicators.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.error("Failed to load maternal indicators.")
    
    with tab2:
        # Update active tab state
        if st.session_state.active_resource_tab != "newborn":
            st.session_state.active_resource_tab = "newborn"
        
        # Load newborn indicators (cached)
        with st.spinner("Loading newborn indicators..."):
            newborn_df = load_newborn_indicators()
        
        if not newborn_df.empty:
            st.markdown("#### 👶 Newborn Health Indicators")
            
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
                background: linear-gradient(135deg, #1565c0, #0d47a1);
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
                background-color: #e3f2fd;
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
                "📥 Download Newborn Indicators",
                data=download_df.to_csv(index=False),
                file_name="newborn_indicators.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.error("Failed to load newborn indicators.")