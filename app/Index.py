import os
import sys
import streamlit as st

project_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="NIR Online-Home", page_icon=":house:", layout="centered")

with open(os.path.join(project_dir, "Index.html"),
          mode = 'r', encoding= 'unicode_escape') as f:
    IndexContents = f.read()
st.markdown(IndexContents, unsafe_allow_html=True)
