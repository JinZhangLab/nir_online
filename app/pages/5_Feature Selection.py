# Import streamlit and other libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pynir.utils import simulateNIR
from pynir.Calibration import pls
from pynir.FeatureSelection import MCUVE, RT, VC
import pandas as pd
from tools.display import plotSPC, plotRef_reg, plotVariableImportance, plotVariableSelection
from tools.dataManipulation import download_csv_md


@st.cache_data
def featureSelection(X, y, options=None):
    methods = options["methods"]
    n_comp_pls = options["n_comp_pls"]
    n_rep = options["n_rep"]
    n_sel = options["n_sel"]

    # Initialize dictionaries to store the selected features and variable importance for each method
    feature_selected = {}
    variable_importance = {}

    if "PLS coefficients" in methods:
        pls_model = pls(n_comp_pls).fit(X.to_numpy(), y.to_numpy())
        feature_selected["PLS coefficients"] = np.argsort(
            np.abs(pls_model.model['B'][1:, n_comp_pls-1]))[::-1][:n_sel]
        variable_importance["PLS coefficients"] = np.abs(
            pls_model.model['B'][1:, n_comp_pls-1])

    if "VIP" in methods:
        pls_model = pls(n_comp_pls).fit(X.to_numpy(), y.to_numpy())
        vip = pls_model.get_vip()
        feature_selected["VIP"] = np.argsort(vip)[::-1][:n_sel]
        variable_importance["VIP"] = vip

    if "MC-UVE" in methods:
        mc_model = MCUVE(X.to_numpy(), y.to_numpy(),
                         n_comp_pls, nrep=n_rep).fit()
        feature_selected["MC-UVE"] = mc_model.featureRank[:n_sel]
        variable_importance["MC-UVE"] = mc_model.criteria

    if "RT" in methods:
        rt_model = RT(X.to_numpy(), y.to_numpy(), n_comp_pls, nrep=n_rep).fit()
        feature_selected["RT"] = rt_model.featureRank[:n_sel]
        variable_importance["RT"] = rt_model.criteria

    if "VC" in methods:
        vc_model = VC(X.to_numpy(), y.to_numpy(), n_comp_pls, nrep=n_rep).fit()
        feature_selected["VC"] = vc_model.featureRank[:n_sel]
        variable_importance["VC"] = vc_model.criteria

    # Create a DataFrame to store the feature selection results
    FS = pd.DataFrame(data=np.full((len(feature_selected),
                      X.shape[1]), False), index=feature_selected.keys(), columns=wv)
    for method in feature_selected.keys():
        FS.loc[method].iloc[feature_selected[method]] = True

    # Create a DataFrame to store the variable importance values
    Imp = pd.DataFrame(variable_importance, index=wv,
                       columns=feature_selected.keys()).T

    if "FS" in locals() and "Imp" in locals() and "X" in locals():
        # Plot the variable selection and variable importance
        col1, col2 = st.columns([1, 1])
        with col1:
            plotVariableSelection(X, FS)
        with col2:
            plotVariableImportance(Imp)

    return FS, Imp


# Create a title and
st.title("Feature Selection for NIR Spectroscopy")
st.markdown("### Upload your data or use our example.")
use_example = st.radio(
    "1.1", ["Example data 1", "Upload data manually"], label_visibility="collapsed")

if use_example == "Example data 1":
    X, y, wv = simulateNIR()
    sampleNames = [f"Sample_{i}" for i in range(X.shape[0])]
    X = pd.DataFrame(X, columns=wv, index=sampleNames)
    y = pd.DataFrame(y, columns=["Reference values"], index=sampleNames)

    st.session_state["X"] = X
    st.session_state["y"] = y

    col1, col2 = st.columns([1, 1])

    with col1:
        plotSPC(X)

    with col2:
        plotRef_reg(y)
elif use_example == "Upload data manually":
    tab1, tab2 = st.columns(2)
    with tab1:
        uploadX = st.file_uploader("Choose file for NIR spectra", type="csv")
        if uploadX is not None:
            X = pd.read_csv(uploadX, index_col=0)
            wv = np.array(X.columns, dtype=float)
            plotSPC(X)
    with tab2:
        uploadY = st.file_uploader(
            "Choose file for reference values", type="csv")
        if uploadY is not None:
            y = pd.read_csv(uploadY, index_col=0)
            plotRef_reg(y)

# Feature selection
if "X" in locals() and "y" in locals():
    # Create a form for method selection
    with st.container():
        st.write("Methods for feature selection:")
        cols = st.columns(2)
        method_pls = cols[0].checkbox("PLS coefficients", value=True)
        method_vip = cols[1].checkbox("VIP", value=True)
        method_mc_uve = cols[0].checkbox("MC-UVE")
        method_rt = cols[1].checkbox("RT")
        method_vc = cols[0].checkbox("VC", value=True)

        # Perform feature selection using different methods
        n_comp_pls = st.slider("Number of components", 1, int(
            np.min(list(X.to_numpy().shape) + [10])), 1)
        n_sel = st.slider("Number of selected features", 1,
                          X.shape[1], round(X.shape[1] * 0.2))
        n_rep = st.slider("Number of repetitions", 1 *
                          X.shape[1], 10*X.shape[1], 2*X.shape[1])

    methods = []
    if method_pls:
        methods.append("PLS coefficients")
    if method_vip:
        methods.append("VIP")
    if method_mc_uve:
        methods.append("MC-UVE")
    if method_rt:
        methods.append("RT")
    if method_vc:
        methods.append("VC")

    options = {"n_comp_pls": n_comp_pls, "n_sel": n_sel,
               "n_rep": n_rep, "methods": methods}

    FS, Imp = featureSelection(X, y, options)

# Apply the feature selection results to other spectral data uplsoaded

if "FS" in locals():
    st.markdown("### Apply Feature Selection results to uploaded Spectra")

    fs_method = st.radio("Apply feature selection",
                         FS.index, label_visibility='collapsed')

    upload_X_new = st.file_uploader(
        "Choose file for NIR spectra to apply feature selection", type="csv")
    if upload_X_new is not None:
        X_new = pd.read_csv(upload_X_new, index_col=0)
        plotSPC(X_new, title="Spectra uploaded")

        X_new_fs = X_new.iloc[:, FS.loc[fs_method].to_numpy()]

        download_csv_md(X_new_fs, fileName="X_fs",
                        label="Download the spectra after feature selection")
