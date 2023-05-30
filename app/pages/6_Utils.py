import streamlit as st
import scipy.io as scio
import numpy as np
import pandas as pd

from pynir.utils import simulateNIR

from tools.display import plotSPC, plotRef
from tools.dataManipulation import download_csv

with st.sidebar:
    functionSel = st.radio(
        "NIR data conversiton.",
        ("Simulate NIR data","Others")
    )



if functionSel == "Simulate NIR data":
    with st.expander("Set simulation parameters"):
        refType = st.radio('Reference value type', ("Continuous","Categorical"))
        n_components = st.slider('Number of samples', 1, 20, 10)
        nSamples = st.slider('Number of samples', 10, 200, 100)
        noiseLevel = st.slider('Noise level (×10^-5)', 1, 100, 1)
        seeds = st.slider('Random seeds', 0, 10000, 0)

    if refType == "Continuous":
        refTypeIdx=1
    elif refType == "Categorical":
        refTypeIdx=2

    X,y, wv = simulateNIR(nSample=nSamples, n_components=n_components,
                          noise = noiseLevel*(10**-5),
                          refType=refTypeIdx,seeds=seeds)
    cols = st.columns(2)
    with cols[0]:
        plotSPC(X=X,wv=wv)
        download_csv(X, label = "Download the spectral file",
                     fileName = "Spectra", columns = wv)

    with cols[1]:
        plotRef(y)
        download_csv(y, label = "Download the reference value file",
                     fileName = "Reference", columns = ["Reference value"])

if functionSel == "Others":
    st.write("Other function  coming soon...")
