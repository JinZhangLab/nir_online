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