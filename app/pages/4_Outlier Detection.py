import pandas as pd
import streamlit as st
import numpy as np
from pynir.utils import simulateNIR
from pynir.OutlierDection import outlierDection_PLS
from tools.display import plotSPC, plotRef_reg

import matplotlib.pyplot as plt

import base64

st.title("Outlier Detection with PLS Regression")
st.write("This app allows you to upload a CSV file with X and y variables and perform outlier detection using PLS regression.")

st.markdown("### Upload your data or use our example.")
use_example = st.radio("Choose an option", ["Example data 1", "Upload data manually"])

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
    od = outlierDection_PLS(ncomp=ncomp, conf=conf)

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
    st.pyplot(fig)

    # Download the outlier detection results
    st.markdown("### Download the outlier detection results")
    df = pd.DataFrame(data=idxOutlier, columns=["Outlier"])
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="OutlierDetection.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
