import pandas as pd
import streamlit as st
import numpy as np
from pynir.utils import simulateNIR
from pynir.OutlierDetection import outlierDetection_PLS
from tools.display import plotSPC, plotRef_reg

import matplotlib.pyplot as plt
from tools.dataManipulation import download_csv_md, download_img


# Page content
st.set_page_config(page_title="NIR Online-Outlier Detection", page_icon=":rocket:", layout="centered")

st.title("Outlier Detection with PLS Regression")
st.write("This app allows you to upload a CSV file with X and y variables and perform outlier detection using PLS regression.")

st.markdown("### Upload your data or use our example.")
use_example = st.radio("Choose an option", ["Example data 1", "Upload data manually"], 
                       key="Outlier_detection_data_selection")

if use_example == "Example data 1":
    X, y, wv = simulateNIR()
    sampleNames = [f"Sample_{i}" for i in range(X.shape[0])]
    X = pd.DataFrame(X, columns=wv, index=sampleNames)
    y = pd.DataFrame(y, columns=["Reference values"], index=sampleNames)

else:
    uploadX = st.file_uploader("Choose file for NIR spectra", type="csv")
    uploadY = st.file_uploader("Choose file for reference values", type="csv")

    if uploadX is not None and uploadY is not None:
        X = pd.read_csv(uploadX, index_col=0)
        y = pd.read_csv(uploadY, index_col=0)

if "X" in locals() and "y" in locals():
    col1, col2 = st.columns([1, 1])

    with col1:
        plotSPC(X)

    with col2:
        plotRef_reg(y)

    ncomp = st.slider("Number of components", 1, int(np.min(list(X.to_numpy().shape) + [10])), 3)
    conf = st.slider("Confidence level", 0.80, 0.99, 0.90)

    # Create an instance of the outlier detection class
    od = outlierDetection_PLS(ncomp=ncomp, conf=conf)

    # Fit the model on X and y
    od.fit(X, y)

    # Detect outliers
    Q, Tsq, Q_conf, Tsq_conf, idxOutlier = od.detect(X, y)

    # Show the number of outliers
    n_outliers = sum(idxOutlier)

    # Plot the Hotelling T2 and Q residuals
    fig, ax = plt.subplots()
    od.plot_HotellingT2_Q(Q, Tsq, Q_conf, Tsq_conf, ax=ax)
    ax.set_title(f"Number of outliers: {n_outliers}")
    
    
    tab1, tab2 = st.tabs(["Figure", "Download"])
    
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="Outlier detection results", label="Download figure")
        download_csv_md(pd.DataFrame(data={"Hotelling T2": Tsq, "Q residuals": Q, "Outlier": idxOutlier}),
                        "Outlier detection results", label="Download results")
