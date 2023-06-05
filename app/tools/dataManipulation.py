# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:44:30 2022

@author: Jin Zhang
"""
import io
import streamlit as st
import pandas as pd
import time
import streamlit as st
import numpy as np

import base64

# @st.cache_data
def convert_fig(fig,figFormat = "pdf"):
    img = io.BytesIO()
    fig.savefig(img, format=figFormat)
    return img

def download_img(fig, fileName = "Figure", label = "Download image"):
    figFormat =  'pdf'
    img = convert_fig(fig,figFormat = figFormat)
    st.download_button(
        label=label,
        data=img,
        file_name= fileName + "." + figFormat,
        mime="image/"+figFormat,
        key=str(time.time())
    )

@st.cache_data
def convert_csv(data, index = True, columns = True, index_label = None):
    return data.to_csv(index=index, header = columns, index_label=index_label).encode('utf-8')
    
def download_csv(data, index = True, columns = True, index_label = None, fileName = "data", label = "Download"):
    csv = convert_csv(data, index = index, columns=columns, index_label=index_label)
    st.download_button(
        label=label,
        data=csv,
        file_name= fileName + '.csv',
        mime='text/csv',
        key=str(time.time())
    )





def download_csv_md(df,index = True, columns = True, index_label = None, fileName = "data", label = "Download"):
  # Convert the dataframe to csv
  csv = df.to_csv(index=True, header=True, index_label=None)
  # Encode the csv to base64
  b64 = base64.b64encode(csv.encode()).decode()
  # Create a markdown link to download the csv
  link = f'<a href="data:file/csv;base64,{b64}" download="{fileName + ".csv"}"> {label} file</a>'
  # Display the link in streamlit
  st.markdown(link, unsafe_allow_html=True)