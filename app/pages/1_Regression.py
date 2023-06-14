import numpy as np
import pandas as pd
import streamlit as st
from pynir.Calibration import pls, regressionReport
from pynir.utils import simulateNIR
from tools.dataManipulation import download_csv
from tools.display import (
    plotPrediction_reg,
    plotRef_reg,
    plotRegressionCoefficients,
    plotSPC,
    plotFOMvsHP,
)
import time

# Step 1: Calibration
def step1():
    st.markdown("## Step 1. Calibration")
    st.markdown("### Upload your data or use our example.")

    use_example = st.radio(
        "Upload your data or use our example.", ["Example data 1", "Upload data manually"],
          on_change=changeReg_state, label_visibility="collapsed", key="reg_data_selection"
    )

    if use_example == "Example data 1":
        # Use example data
        X, y, wv = simulateNIR()
        sampleNames = [f"Sample_{i}" for i in range(X.shape[0])]
        X = pd.DataFrame(X, columns=wv, index=sampleNames)
        y = pd.DataFrame(y, columns=["Reference values"], index=sampleNames)

        col1, col2 = st.columns([1, 1])

        with col1:
            plotSPC(X)

        with col2:
            plotRef_reg(y)

    elif use_example == "Upload data manually":
        # Upload data manually
        st.info(
            "The spectral file you upload needs to meet the requirements such that (1) each row is a spectrum of a sample, (2) the first row is the wavelength, and (3) the first column is the name of the sample.",
            icon="ℹ️",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload your spectra here for calibration", "csv", key="reg_Xcal"+str(st.session_state.reg))
            if uploaded_file is not None:
                X = pd.read_csv(uploaded_file, index_col=0)
                wv = np.array(X.columns).astype("float")
                sampleNames = X.index

                plotSPC(X)

        with col2:
            uploaded_file = st.file_uploader("Upload your reference data here for calibration", "csv", key="reg_ycal"+str(st.session_state.reg))
            if uploaded_file is not None:
                y = pd.read_csv(uploaded_file, index_col=0)
                plotRef_reg(y)
    if "X" in locals() and "y" in locals():
        return X, y

# Step 2: Cross validation
def step2(X, y):
    st.markdown("## Step 2. Cross validation")
    with st.container():
        st.markdown("### Set the parameters for cross validation.")
        n_components = st.slider("The max number of pls components in cross validation.", 1, 20, 10)
        nfold = st.slider("The number of fold in cross validation.", 2, 20, 10)

    plsModel = pls(n_components=n_components)
    plsModel.fit(X.to_numpy(), y.to_numpy())

    yhat = plsModel.predict(X.to_numpy(), n_components=np.arange(n_components) + 1)
    yhat_cv = plsModel.crossValidation_predict(nfold)
    rmsec = [regressionReport(y.to_numpy(), yhat[:, i])["rmse"] for i in range(yhat_cv.shape[1])]
    r2 = [regressionReport(y.to_numpy(), yhat[:, i])["r2"] for i in range(yhat_cv.shape[1])]
    rmsecv = [regressionReport(y.to_numpy(), yhat_cv[:, i])["rmse"] for i in range(yhat_cv.shape[1])]
    r2cv = [regressionReport(y.to_numpy(), yhat_cv[:, i])["r2"] for i in range(yhat_cv.shape[1])]

    col1, col2 = st.columns([1, 1])
    with col1:
        RMSECV = pd.DataFrame(
            data=[rmsec, rmsecv], index=["RMSE", "RMSECV"], columns=np.arange(n_components) + 1
        )
        plotFOMvsHP(RMSECV, xlabel="$n$LV", ylabel="RMSE", title="RMSE vs $n$LV")
    with col2:
        R2cv = pd.DataFrame(data=[r2, r2cv], index=["R2", "R$^2$$_C$$_V$"], columns=np.arange(n_components) + 1)
        plotFOMvsHP(R2cv, xlabel="$n$LV", ylabel="R2", title="R$^2$ vs $n$LV")

    st.markdown("### Set the optimal number of component.")
    optLV = st.slider("optLV", 1, n_components, int(np.argmin(rmsecv) + 1))
    plsModel.optLV = optLV

    col1, col2 = st.columns([1, 1])
    with col1:
        modelCoefficients = pd.DataFrame(
            data=plsModel.model["B"][:, optLV - 1].reshape(1, -1),
            index=["Coefficients"],
            columns=[-1] + list(X.columns),
        )
        plotRegressionCoefficients(modelCoefficients, title=f"Regression Coefficients with {optLV} LVs")
    with col2:
        ycv = pd.DataFrame(
            data=np.column_stack([y.to_numpy().ravel(), yhat[:, optLV - 1], yhat_cv[:, optLV - 1]]),
            columns=["Reference", "Calibration", "Cross Validation"],
            index=y.index,
        )
        plotPrediction_reg(ycv, xlabel="Reference", ylabel="Prediction", title=f"Prediction with {optLV} LVs")
    
    if "plsModel" in locals():
        return plsModel

# Step 3: Prediction
def step3(plsModel):
    st.markdown("## Step 3. Prediction.")

    st.markdown("### Import your NIR spectra data for prediction")
    uploaded_file = st.file_uploader("X for prediction_reg", "csv", label_visibility="hidden", key="reg_Xtest"+str(st.session_state.reg))
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        wv = np.array(X.columns).astype("float")
        sampleName = X.index

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X)

        yhat = plsModel.predict(X.to_numpy(), n_components=plsModel.optLV)
        yhat = pd.DataFrame(data=yhat, index=sampleName, columns=["Prediction"])
        with col2:
            st.markdown("### Prediction results")
            st.dataframe(yhat)
            download_csv(yhat, index=True, columns=True, index_label="Sample Name",
                fileName="Prediction", label="Download Prediction")

        st.markdown("### Import your reference values for visualization")
        uploaded_file = st.file_uploader("y for prediction_reg", "csv", label_visibility="hidden", key="reg_ytest"+str(st.session_state.reg))
        if uploaded_file is not None:
            y = pd.read_csv(uploaded_file, index_col=0)
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                Predictions = pd.DataFrame(data=np.column_stack([y.to_numpy().flatten(), yhat.to_numpy().flatten()]),
                    columns=["Reference", "Prediction"], index=y.index)
                plotPrediction_reg(Predictions, xlabel="Reference", ylabel="Prediction", title="Predictions" )


def changeReg_state():
    st.session_state.reg += 1

# Page content
st.set_page_config(page_title="NIR Online-Regression", page_icon=":eyeglasses:", layout="centered")

if 'reg' not in st.session_state:
    st.session_state.reg = 0

st.markdown("""
            # Regression for NIR spectra
            ---
            Commonly, the regression of NIR spectra includes three steps: 
            (1) Calibration, 
            (2) Cross validation for determining the optimal calibration hyper-parameters, 
            (3) Prediction on new measured spectra. \n
            The spectra (X) and reference values (y) file accepted by tools in this site only support  csv format, with sample in rows, and wavelength in columns. 
            The first row contains the wavelength, and the first column is the sample name. If your data is not in the supported format, convert it at the utils page on our site. 
            """)

results1 = step1()
if results1 is not None:
    X, y = results1

if 'X' in locals() and "y" in locals() and  X.shape[0] == len(y):
    plsModel = step2(X, y)

if 'plsModel' in locals():
    if plsModel is not None:
        step3(plsModel)
