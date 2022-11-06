import streamlit as st
import scipy.io as scio
import numpy as np
import pandas as pd

from pynir.utils import simulateNIR

from tools.display import plotSPC, plotRef


with st.sidebar:
    dataload_type = st.radio(
        "NIR data conversiton.",
        ("Simulate NIR data","Others")
    )



if dataload_type == "Simulate NIR data":
    with st.expander("Set simulation parameters"):
        refType = st.radio('Reference value type', ("Continuous","categorical"))
        nComponents = st.slider('Number of samples', 1, 20, 10)
        nSamples = st.slider('Number of samples', 10, 200, 100)
        noiseLevel = st.slider('Noise level (×10^-5)', 1, 100, 1)
        seeds = st.slider('Random seeds', 0, 10000, 0)
    
    if refType == "Continuous":
        refTypeIdx=1
    elif refType == "categorical":
        refTypeIdx=2
    
    X,y, wv = simulateNIR(nSample=nSamples, nComp=nComponents,
                          noise = noiseLevel*(10**-5), 
                          refType=refTypeIdx,seeds=seeds)
    plotSPC(X=X,wv=wv)
    plotRef(y)
    
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return pd.DataFrame(df).to_csv(header=False, index=False).encode('utf-8')

    Xcsv = convert_df(X)

    st.download_button(
        label="Download simulated spectra as CSV",
        data=Xcsv,
        file_name='X.csv',
        mime='text/csv',
    )

    ycsv = convert_df(y)

    st.download_button(
        label="Download simulated reference values as CSV",
        data=ycsv,
        file_name='y.csv',
        mime='text/csv',
    )

    

