import streamlit as st

from pynir.Preprocessing import cwt, snv, msc, SG_filtering
from pynir.utils import simulateNIR

import numpy as np
import pandas as pd

from tools.display import plotSPC
from tools.dataManipulation import download_csv


## Dependent function
def dataProcessing_cwt(X):
    st.markdown("Set the parameters")
    scale = st.slider("scale",1,X.shape[1],30)
    wavelet = st.radio("Wavelet",["cgau1","cgau2","cgau3","cgau4","cgau5","cgau6",
                                  "cgau7","cgau8","cmor","fbsp", "gaus1","gaus2",
                                  "gaus3","gaus4","gaus5","gaus6","gaus7","gaus8",
                                  "mexh","morl","shan"],
                       horizontal=True)

    cwtModel = cwt(wavelet = wavelet, scale = scale)
    return cwtModel

def dataProcessing_snv(X):
    snvModel = snv().fit(X)
    return snvModel

def dataProcessing_msc(X):
    mscModel = msc().fit(X)
    return mscModel

def dataProcessing_sg_smooth(X):
    st.markdown("Set the parameters")
    window_length = st.slider("window length",1,min((*X.shape,50)),7)
    polyorder = st.slider("polyorder",0,3,1)
    sgModel = SG_filtering(window_length = window_length,
                           polyorder=polyorder,deriv = 0)
    return sgModel

def dataProcessing_sg_derivate(X):
    st.markdown("Set the parameters")
    window_length = st.slider("window length",1,min((*X.shape,50)),7)
    polyorder = st.slider("polyorder",0, 3, 1)
    deriv = st.slider("derivate order", 0, polyorder, 0)
    sgModel = SG_filtering(window_length = window_length,
                           polyorder=polyorder, deriv = deriv)
    return sgModel



## Main Page
st.markdown("# Spectral preprocessing")

dataSource = st.radio("Upload your data or use our example.",
                      ["Example data 1", "Upload data manually"])

method = st.radio("Select a preprocessing method",
                  ["CWT","SNV","MSC","SG_Smooth","SG_Derivate"],
                  horizontal=True)

if dataSource == "Example data 1":
    X, y, wv = simulateNIR()
elif dataSource == "Upload data manually":
    uploaded_file = st.file_uploader("Upload your spectra here","csv")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file,index_col=0)
        wv = np.array(X.columns).astype("float")
        sampleNames = X.index
        X = np.array(X)


if "X" in list(locals().keys()):
    if method == "CWT":
        ppModel = dataProcessing_cwt(X)

    elif method == "SNV":
        ppModel = dataProcessing_snv(X)

    elif method == "MSC":
        ppModel = dataProcessing_msc(X)

    elif method == "SG_Smooth":
        ppModel = dataProcessing_sg_smooth(X)

    elif method == "SG_Derivate":
        ppModel = dataProcessing_sg_derivate(X)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### NIR spectra uploaded.")
        plotSPC(X=X,wv = wv)
    with col2:
        st.markdown("### NIR spectra Preprocessed.")
        plotSPC(X=ppModel.transform(X),wv = wv)
        download_csv(ppModel.transform(X), label = "Download the preprocessed spectral file",
                     fileName = "Spectra_preprocessed", columns = wv)

if "ppModel" in list(locals().keys()):
    uploaded_file_new = st.file_uploader("Preprecess spectra with the selected parameters","csv")
    if uploaded_file_new is not None:
        Xnew = pd.read_csv(uploaded_file_new,index_col=0)
        wv_new = np.array(Xnew.columns).astype("float")
        sampleNames_new = Xnew.index
        Xnew = np.array(Xnew)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### NIR spectra uploaded.")
            plotSPC(X=Xnew,wv = wv)
        with col2:
            st.markdown("### NIR spectra Preprocessed.")
            plotSPC(X=ppModel.transform(Xnew),wv = wv)
            download_csv(ppModel.transform(Xnew), label = "Download the preprocessed spectral file",
                         fileName = "Spectra_preprocessed", columns = wv)
