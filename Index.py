import streamlit as st
with open("Index.html",'r', encoding= 'unicode_escape') as f:
    IndexContents = f.read()
st.markdown(IndexContents, unsafe_allow_html=True)
