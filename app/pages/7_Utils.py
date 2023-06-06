import streamlit as st
import scipy.io as scio
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from pynir.Calibration import sampleSplit_KS
from pynir.utils import simulateNIR

from tools.display import plotSPC, plotRef_reg, plotRef_clf, pltPCAscores_2d
from tools.dataManipulation import download_csv

from sklearn.model_selection import train_test_split

import time

def fun_simulateNIR():
    with st.container():
        st.markdown("### Set simulation parameters")
        n_components = st.slider('Number of components', 2, 20, 10)
        nSamples = st.slider('Number of samples', 10, 2000, 100)
        noiseLevel = st.slider('Noise level (×10^-5)', 0, 100, 1)
        seeds = st.slider('Random seeds', 0, 10000, 0)
        refType = st.slider('Reference value type', 1, min([5, round(nSamples/2)]), 1,
                            help="""1 represent for reference values resampled from contious regrion, 
                           integer larger than 1 for reference values belong to the corresponding number of classes.""")

    X, y, wv = simulateNIR(nSample=nSamples,
                           n_components=n_components,
                           noise=noiseLevel*(10**-5),
                           refType=refType, seeds=seeds)
    sampleName = [f"Sample_{i}" for i in range(nSamples)]
    X = pd.DataFrame(X, index=sampleName, columns=wv)
    y = pd.DataFrame(y, index=sampleName, columns=["Reference value"])
    cols = st.columns(2)
    with cols[0]:
        plotSPC(X)

    with cols[1]:
        if refType == 1:
            plotRef_reg(y)
        else:
            plotRef_clf(y)


def fun_sampleSplit():
    st.markdown("""
    ## Split sample set into calibration and validation set
    Two methods are supportted to split sample set now, random and Kennard-Stone (KS) algorithm.
    """)
    split_method = st.radio("Split method", ("Random", "KS"), on_change=st.cache_data.clear())
    
    if split_method == "Random":
        num_samples = st.slider("Number of samples", 10, 2000, 100)
        split_ratio = st.slider("Train ratio", 0.1, 0.9, 0.8)
        seeds = st.slider("Random seeds", 0, 10000, 0)

        sampleIdx = np.arange(num_samples)
        sampleName = np.array([f"Sample_{i}" for i in range(num_samples)])
        trainIdx, testIdx = train_test_split(sampleIdx,
                                             test_size = round(num_samples*(1-split_ratio)),
                                             random_state = seeds,
                                             shuffle = True)
        trainIdx = pd.DataFrame(data = trainIdx, index = sampleName[trainIdx], columns = ["Train set index"])
        testIdx = pd.DataFrame(data = testIdx, index = sampleName[testIdx], columns = ["Test set index"])
        col1, col2 = st.columns(2)
        with col1:
            st.write(trainIdx)
        with col2:
            st.write(testIdx)
    elif split_method == "KS":
        st.markdown("""
        The KS algorithms aim to select a subset of samples whose spectra are as different from each other as possible.
        Therefore, your spectra should be uploaded as a csv file with sample in rows and wavenumber in columns.
        """)
        uploaded_file = st.file_uploader("Upload your spectra here", "csv")
        if uploaded_file is not None:
            X = pd.read_csv(uploaded_file, index_col=0)
            sampleName = X.index.to_numpy(dtype=str)
            wv = X.columns.to_numpy(dtype=float)

            scores = PCA(n_components=2).fit_transform(X)
            scores = pd.DataFrame(data=scores, index=sampleName, columns=["PC1", "PC2"])
            pltPCAscores_2d(scores=scores, title="PCA scores of samples")

            split_ratio = st.slider("Train ratio", 0.1, 0.9, 0.8)
            trainIdx, testIdx = sampleSplit_KS(X.to_numpy(), test_size=1-split_ratio)
            trainIdx = pd.DataFrame(data = trainIdx, index = sampleName[trainIdx], columns = ["Train set index"])
            testIdx = pd.DataFrame(data = testIdx, index = sampleName[testIdx], columns = ["Test set index"])
            col1, col2 = st.columns(2)
            with col1:
                st.write(trainIdx)
            with col2:
                st.write(testIdx)

    if "trainIdx" in locals() and "testIdx" in locals():
        st.markdown("## Apply the split to your spectral or reference value data: ")
        uploaded_file = st.file_uploader("Upload file to be split here", "csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, index_col=0)
            col1, col2 = st.columns(2)
            with col1:
                download_csv(data.iloc[trainIdx.iloc[:,0],:], index=True, columns=True, fileName="trainSet", label="Download train set")
            with col2:
                download_csv(data.iloc[testIdx.iloc[:,0],:], index=True, columns=True, fileName="testSet", label="Download test set")

def fun_convertWN2WL():
    st.markdown("""
    Convert the wavenumber in a spectral file with WaveNumber (cm-1) unit to Wavelength (nm).
    Before conversion, be sure that the fisrt row of your spectral file is the wavenumber with unit of cm^-^1.
    """)
    uploaded_file = st.file_uploader("Upload your spectra here", "csv")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        sampleName = X.index.to_numpy(dtype=str)
        wv = X.columns.to_numpy(dtype=float)
        wv = 1e7/wv
        X = pd.DataFrame(data=X.to_numpy(), index=sampleName, columns=wv)
        download_csv(X, index=True, columns=True, index_label="Sample Name\\Wavelength (nm)", fileName="Spectra_converted", label="Download converted spectra")

def fun_transpose():
    st.markdown("""
    You can use this function to transpose the row and column in a csv file
    """)
    uploaded_file = st.file_uploader("Upload your file here", "csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0)
        data = data.transpose()
        download_csv(data, index=True, columns=True, fileName="Data_transposed", label="Download transposed data")

# Page content
st.set_page_config(page_title="NIR Online-Utils",
                   page_icon=":rocket:", layout="wide")
st.markdown("# Tools collection for NIR calibration")

functionSel = st.radio(
    "Tool collection may helpful for NIR calibation.",
    ("Simulate NIR data",
     "Split Sample Set",
     "Convert WaveNumber (cm-1) to Wavelength (nm)",
     "Transpose data in csv file",
     "Others"),
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

if functionSel == "Simulate NIR data":
    fun_simulateNIR()

if functionSel == "Split Sample Set":
    fun_sampleSplit()

if functionSel == "Convert WaveNumber (cm-1) to Wavelength (nm)":
    fun_convertWN2WL()

if functionSel == "Transpose data in csv file":
    fun_transpose()

if functionSel == "Others":
    st.write("Other function  coming soon...")
