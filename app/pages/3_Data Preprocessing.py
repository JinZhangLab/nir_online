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
    wavelet = st.radio("Wavelet",["gaus1","gaus2", "mexh","morl","shan"],
                       horizontal=True)

    cwtModel = cwt(wavelet = wavelet, scale = scale)
    return cwtModel

def dataProcessing_snv(X):
    snvModel = snv()
    snvModel.fit(X)
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
    polyorder = st.slider("polyorder",0, 3, 2)
    deriv = st.slider("derivate order", 0, polyorder, 0)
    sgModel = SG_filtering(window_length = window_length,
                           polyorder=polyorder, deriv = deriv)
    return sgModel



# page content
st.set_page_config(page_title="NIR Online-Data preprocessing", page_icon=":rocket:", layout="wide")

st.markdown("# Spectral preprocessing")

dataSource = st.radio("Upload your data or use our example.",
                      ["Example data 1", "Upload data manually"])

method = st.radio("Select a preprocessing method",
                  ["CWT","SNV","MSC","SG_Smooth","SG_Derivate"],
                  horizontal=True)

if dataSource == "Example data 1":
    X, y, wv = simulateNIR()
    sampleNames = [f"Sample_{i}" for i in range(X.shape[0])]
    X = pd.DataFrame(X, columns=wv, index=sampleNames)
    y = pd.DataFrame(y, columns=["Reference values"], index=sampleNames)

elif dataSource == "Upload data manually":
    uploaded_file = st.file_uploader("Upload your spectra here","csv")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file,index_col=0)
        wv = np.array(X.columns).astype("float")
        sampleNames = X.index



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
        plotSPC(X)
    with col2:
        st.markdown("### NIR spectra Preprocessed.")
        X_preprocessed = ppModel.transform(X.to_numpy())
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=wv, index=sampleNames)
        plotSPC(X_preprocessed)

    if "ppModel" in list(locals().keys()):
        st.markdown("## Apply the preprocess model to another spectral set")
        uploaded_file_new = st.file_uploader("Apply the preprocess model to another spectral set","csv", label_visibility='collapsed')
        if uploaded_file_new is not None:
            Xnew = pd.read_csv(uploaded_file_new,index_col=0)
            wv_new = np.array(Xnew.columns).astype("float")
            if np.sum(wv_new != wv) > 0:
                st.error("The wavelength range of the uploaded spectra is not the same as the uploaded spectra.")
                st.stop()

            sampleNames_new = Xnew.index

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### NIR spectra uploaded.")
                plotSPC(Xnew)
            with col2:
                st.markdown("### NIR spectra Preprocessed.")
                Xnew_preprocessed = ppModel.transform(Xnew.to_numpy())
                Xnew_preprocessed = pd.DataFrame(Xnew_preprocessed, columns=wv, index=sampleNames_new)
                plotSPC(Xnew_preprocessed)

