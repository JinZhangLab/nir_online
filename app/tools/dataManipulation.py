# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:44:30 2022

@author: chinn
"""
import io
import streamlit as st
import pandas as pd
import time
import streamlit as st
import numpy as np

#@st.cache
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
def convert_csv(data,columnName = None):
    if not isinstance(data, pd.core.frame.DataFrame):
        data = pd.DataFrame(data = data, columns=columnName)
    return data.to_csv(header=True, index=True).encode('utf-8')
    
def download_csv(data, fileName = "data", label = "Download",
                 columns = None):
    csv = convert_csv(data, columnName=columns)
    st.download_button(
        label=label,
        data=csv,
        file_name= fileName + '.csv',
        mime='text/csv',
    )


