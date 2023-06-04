import numpy as np
import pandas as pd
import streamlit as st
from pynir.Calibration import pls, regressionReport
from pynir.utils import simulateNIR
from tools.dataManipulation import download_csv
from tools.display import (plotPrediction_reg, plotRef_reg,
                           plotRegressionCoefficients, plotSPC, plotFOMvsHP)
import time

def step1():
    st.markdown("## Step 1. Calibration")
    st.markdown("### Upload your data or use our example.")
    use_example = st.radio("1.1", ["Example data 1", "Upload data manually"], 
                           on_change=st.cache_data.clear(), label_visibility="collapsed")

    try:
        del st.session_state.X, st.session_state.y, st.session_state.plsModel
    except:
        pass

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
        st.info("""
                The spectral file you upload needs to meet the requirements
                such that (1) each row is a spectrum of a sample, (2) the first row
                is the wavelength, and (3) the first column is the name of the sample.
                """, icon="ℹ️")

        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload your spectra here", "csv")
            if uploaded_file is not None:
                X = pd.read_csv(uploaded_file, index_col=0)
                wv = np.array(X.columns).astype("float")
                sampleNames = X.index

                plotSPC(X)
                st.session_state["X"] = X

        with col2:
            uploaded_file = st.file_uploader("Upload your reference data here", "csv")
            if uploaded_file is not None:
                y = pd.read_csv(uploaded_file, index_col=0)
                plotRef_reg(y)
                st.session_state["y"] = y


def step2():
    st.markdown("## Step 2. Cross validation")

    X = st.session_state["X"]
    y = st.session_state["y"]


    with st.container():
        st.markdown("### Set the parameters for cross validation.")
        n_components = st.slider("The max number of pls components in cross validation.", 1, 20, 10)
        nfold = st.slider("The number of fold in cross validation.", 2, 20, 10)

    plsModel = pls(n_components=n_components)
    plsModel.fit(X.to_numpy(), y.to_numpy())
    st.session_state["plsModel"] = plsModel

    yhat = plsModel.predict(X.to_numpy(), n_components=np.arange(n_components) + 1)
    yhat_cv = plsModel.crossValidation_predict(nfold)
    rmsec = [regressionReport(y.to_numpy(), yhat[:, i])["rmse"] for i in range(yhat_cv.shape[1])]
    r2 = [regressionReport(y.to_numpy(), yhat[:, i])["r2"] for i in range(yhat_cv.shape[1])]
    rmsecv = [regressionReport(y.to_numpy(), yhat_cv[:, i])["rmse"] for i in range(yhat_cv.shape[1])]
    r2cv = [regressionReport(y.to_numpy(), yhat_cv[:, i])["r2"] for i in range(yhat_cv.shape[1])]

    col1, col2 = st.columns([1, 1])
    with col1:
        RMSECV = pd.DataFrame(data=[rmsec, rmsecv], index=["RMSE", "RMSECV"], columns=np.arange(n_components) + 1)
        plotFOMvsHP(RMSECV, xlabel="$n$LV", ylabel="RMSE", title="RMSE $vs$ $n$LV")
    with col2:
        R2cv = pd.DataFrame(data=[r2, r2cv], index=["R2", "R$^2$$_C$$_V$"], columns=np.arange(n_components) + 1)
        plotFOMvsHP(R2cv, xlabel="$n$LV", ylabel="R2", title="R$^2$ $vs$ $n$LV")
    
    st.markdown("### Set the optimal number of component.")
    optLV = st.slider("optLV", 1, n_components, int(np.argmin(rmsecv) + 1))
    st.session_state["plsModel"].optLV = optLV

    col1, col2 = st.columns([1, 1])
    with col1:
        modelCoefficients = pd.DataFrame(data=plsModel.model["B"][:, optLV - 1], index=[-1]+ list(X.columns), 
                                         columns=["Coefficients"])
        plotRegressionCoefficients(modelCoefficients, title=f"Regression Coefficients with {optLV} LVs")
    with col2:
        ycv = pd.DataFrame(data = [y.to_numpy().flatten(), yhat[:,optLV-1], yhat_cv[:, optLV - 1]],
                            index=["Reference", "Calibration","Cross Validation"], columns=y.index)
        plotPrediction_reg(ycv,xlabel="Reference", ylabel="Prediction", title=f"Prediction with {optLV} LVs")


def step3():
    plsModel = st.session_state["plsModel"]
    st.markdown("## Step 3. Prediction.")

    st.markdown("### Import your NIR spectra data for prediction")
    uploaded_file = st.file_uploader("X for prediction", "csv", label_visibility="hidden")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        wv = np.array(X.columns).astype("float")
        sampleName = X.index

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X)

        yhat = plsModel.predict(X.to_numpy(), n_components=plsModel.optLV)
        yhat = pd.DataFrame(data=yhat,index = sampleName, columns=["Prediction"])
        with col2:
            st.markdown("### Prediction results")
            st.dataframe(yhat)
            download_csv(yhat, index = True, columns=True, index_label="Sample Name",
                          fileName="Prediction", label="Download Prediction")

        st.markdown("### Import your reference values for visualization")
        uploaded_file = st.file_uploader("y for prediction", "csv", label_visibility="hidden")
        if uploaded_file is not None:
            y = pd.read_csv(uploaded_file, index_col=0)
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                Predictions = pd.DataFrame(data = [y.to_numpy().flatten(), yhat.to_numpy().flatten()],
                    index=["Reference", "Prediction"], columns=y.index)
                plotPrediction_reg(Predictions,xlabel="Reference", ylabel="Prediction", title="Predictions")


# page content
st.set_page_config(page_title="NIR Online-Regression", page_icon=":eyeglasses:", layout="wide")
st.markdown('---')
st.markdown("""
            # Regression for NIR spectra
            Commonly, the regression of NIR spectra include three steps: 
            (1) Calibration, 
            (2) Cross validation for determing the optimal calibration hyper-parameters, 
            (3) Prediction on new measured spectra. \n
            The spectra (X) and reference values (y) file accepted by tools in this site only support  csv format, with sample in rows, and wavelength in columns. 
            The first row contains the wavelength, and first column is the sample name. If your data are not in the supportted format, convert it at the utils page in our site. 
            """)

step1()
if 'X' in st.session_state and 'y' in st.session_state and st.session_state["X"].shape[0] == len(st.session_state["y"]):
    step2()
if 'plsModel' in st.session_state:
    step3()
