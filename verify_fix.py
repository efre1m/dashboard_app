import pandas as pd
import sys
import os

# Mock streamlit since we're running in a script
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
    def error(self, msg): print(f"ST ERROR: {msg}")
    def info(self, msg): print(f"ST INFO: {msg}")
    def subheader(self, msg): print(f"ST SUBHEADER: {msg}")
    def radio(self, *args, **kwargs): return "line"
    def columns(self, n): return [MockCol() for _ in range(n)]
    def metric(self, *args, **kwargs): pass
    def plotly_chart(self, *args, **kwargs): print("CHART RENDERED")
    def expander(self, *args, **kwargs): return MockExpander()
    def dataframe(self, *args, **kwargs): print("DF RENDERED")
    def download_button(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass

class MockCol:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def metric(self, *args, **kwargs): pass

class MockExpander:
    def __enter__(self): return self
    def __exit__(self, *args): pass

import streamlit as st
sys.modules['streamlit'] = MockStreamlit()

# Add project root to path
sys.path.append(os.getcwd())

from utils.kpi_utils import render_trend_chart

# Create sample data for Admitted Mothers
df = pd.DataFrame({
    'period_display': ['Jan-24', 'Feb-24'],
    'value': [10.0, 15.0],
    'numerator': [10, 15],
    'denominator': [1, 1]
})

print("Testing render_trend_chart for Admitted Mothers...")
try:
    render_trend_chart(
        df=df,
        period_col='period_display',
        value_col='value',
        title='Total Admitted Mothers Trend',
        bg_color='#FFFFFF',
        text_color='#000000',
        facility_names=['Test Facility'],
        numerator_name='Admitted Mothers',
        facility_uids=['UID1']
    )
    print("SUCCESS: render_trend_chart finished without crashing.")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
