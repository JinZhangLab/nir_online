import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pynir.Calibration import sampleSplit_KS
from pynir.utils import simulateNIR
from tools.display import plotSPC, plotRef_reg, plotRef_clf, pltPCAscores_2d
from tools.dataManipulation import download_csv
from sklearn.model_selection import train_test_split

# Simulate NIR Data
def simulate_nir_data():
    st.markdown("### Set simulation parameters")
    n_components = st.slider('Number of components', 2, 20, 10)
    n_samples = st.slider('Number of samples', 10, 2000, 100)
    noise_level = st.slider('Noise level (×10^-5)', 0.00, 10.00, 1.00)
    seeds = st.slider('Random seeds', 0, 10000, 0)
    ref_type = st.slider('Reference value type', 1, min([5, round(n_samples/2)]), 1,
                         help="""1 represents reference values resampled from continuous region,
                                 integer larger than 1 represents reference values belonging to the corresponding number of classes.""")

    X, y, wv = simulateNIR(nSample=n_samples,
                           n_components=n_components,
                           noise=noise_level*(10**-5),
                           refType=ref_type, seeds=seeds)
    sampleName = [f"Sample_{i}" for i in range(n_samples)]
    X = pd.DataFrame(X, index=sampleName, columns=wv)
    y = pd.DataFrame(y, index=sampleName, columns=["Reference value"])
    cols = st.columns(2)
    with cols[0]:
        plotSPC(X)

    with cols[1]:
        if ref_type == 1:
            plotRef_reg(y)
        else:
            plotRef_clf(y)

# Split Sample Set
def split_sample_set():
    st.markdown("""
        ## Split sample set into calibration and validation set
        Two methods are supported to split the sample set now: random and Kennard-Stone (KS) algorithm.
    """)
    split_method = st.radio("Split method", ("Random", "KS"), on_change=st.cache_data.clear())

    if split_method == "Random":
        num_samples = st.slider("Number of samples", 10, 2000, 100)
        split_ratio = st.slider("Train ratio", 0.1, 0.9, 0.8)
        seeds = st.slider("Random seeds", 0, 10000, 0)

        sampleIdx = np.arange(num_samples)
        sampleName = np.array([f"Sample_{i}" for i in range(num_samples)])
        trainIdx, testIdx = train_test_split(sampleIdx,
                                             test_size=round(num_samples * (1 - split_ratio)),
                                             random_state=seeds,
                                             shuffle=True)
        trainIdx = pd.DataFrame(data=trainIdx, index=sampleName[trainIdx], columns=["Train set index"])
        testIdx = pd.DataFrame(data=testIdx, index=sampleName[testIdx], columns=["Test set index"])
        col1, col2 = st.columns(2)
        with col1:
            st.write(trainIdx)
        with col2:
            st.write(testIdx)
    elif split_method == "KS":
        st.markdown("""
            The KS algorithms aim to select a subset of samples whose spectra are as different from each other as possible.
            Therefore, your spectra should be uploaded as a CSV file with samples in rows and wavenumber in columns.
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
            trainIdx, testIdx = sampleSplit_KS(X.to_numpy(), test_size=1 - split_ratio)
            trainIdx = pd.DataFrame(data=trainIdx, index=sampleName[trainIdx], columns=["Train set index"])
            testIdx = pd.DataFrame(data=testIdx, index=sampleName[testIdx], columns=["Test set index"])
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
                    download_csv(data.iloc[trainIdx.iloc[:, 0], :], index=True, columns=True, fileName="trainSet",
                                 label="Download train set")
                with col2:
                    download_csv(data.iloc[testIdx.iloc[:, 0], :], index=True, columns=True, fileName="testSet",
                                 label="Download test set")

# Convert WaveNumber (cm-1) to Wavelength (nm)
def convert_wn_to_wl():
    st.markdown("""
        Convert the wavenumber in a spectral file with WaveNumber (cm-1) unit to Wavelength (nm).
        Before conversion, be sure that the first row of your spectral file is the wavenumber with the unit of cm^-^1.
    """)
    uploaded_file = st.file_uploader("Upload your spectra here", "csv")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        sampleName = X.index.to_numpy(dtype=str)
        wv = X.columns.to_numpy(dtype=float)
        wv = 1e7 / wv
        X = pd.DataFrame(data=X.to_numpy(), index=sampleName, columns=wv)
        download_csv(X, index=True, columns=True, index_label="Sample Name\\Wavelength (nm)",
                     fileName="Spectra_converted", label="Download converted spectra")

# Transpose data in CSV file
def transpose_data():
    st.markdown("""
        You can use this function to transpose the row and column in a CSV file.
    """)
    uploaded_file = st.file_uploader("Upload your file here", "csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0)
        data = data.transpose()
        download_csv(data, index=True, columns=True, fileName="Data_transposed", label="Download transposed data")

# Page content
st.set_page_config(page_title="NIR Online-Utils", page_icon=":rocket:", layout="wide")
st.markdown("# Tools collection for NIR calibration")

function_sel = st.radio(
    "Tool collection that may be helpful for NIR calibration.",
    ("Simulate NIR data",
     "Split Sample Set",
     "Convert WaveNumber (cm-1) to Wavelength (nm)",
     "Transpose data in CSV file",
     "Others"),
    horizontal=True,
    key="function_selector"
)

st.markdown("---")

if function_sel == "Simulate NIR data":
    simulate_nir_data()

if function_sel == "Split Sample Set":
    split_sample_set()

if function_sel == "Convert WaveNumber (cm-1) to Wavelength (nm)":
    convert_wn_to_wl()

if function_sel == "Transpose data in CSV file":
    transpose_data()

if function_sel == "Others":
    st.write("Other functions coming soon...")
