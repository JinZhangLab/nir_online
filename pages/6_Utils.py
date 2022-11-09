import streamlit as st
import scipy.io as scio
import numpy as np
import pandas as pd

from pynir.utils import simulateNIR

from tools.display import plotSPC, plotRef
from tools.dataManipulation import download_csv

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
    download_csv(X, label = "Download the spectral file", 
                 fileName = "Spectra", columns = wv)  
    
    plotRef(y)
    download_csv(y, label = "Download the reference value file", 
                 fileName = "Reference", columns = ["Reference value"])

if dataload_type == "Others":
    st.write("Other function  coming soon...")        

    

