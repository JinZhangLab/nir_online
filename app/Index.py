import streamlit as st

st.set_page_config(page_title="NIR Online-Home", page_icon=":house:", layout="wide")

with open("Index.html",'r', encoding= 'unicode_escape') as f:
    IndexContents = f.read()
st.markdown(IndexContents, unsafe_allow_html=True)
