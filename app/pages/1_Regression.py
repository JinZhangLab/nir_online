import streamlit as st

import numpy as np
import pandas as pd

from pynir.Calibration import pls, regresssionReport
from pynir.utils import simulateNIR

from tools.display import (plotSPC, plotRef, plotRMSECV, plotR2CV,
                     plotPredictionCV,plotPrediction, plotRegressionCoefficients)
from tools.dataManipulation import download_csv

def step1():
    st.markdown("## 1. Calibration")
    st.markdown('---')

    use_exapmle = st.radio("Upload your data or use our example.",["Example data 1","Upload data manually"])


    if use_exapmle == "Example data 1":
        try:
            del st.session_state.X, st.session_state.y, st.session_state.plsModel
        except:
            pass
        X, y,wv = simulateNIR()
        st.session_state["X"]=X
        st.session_state["y"]=y
        col1, col2 = st.columns([1,1])
        with col1:
            plotSPC(X=X, wv = wv)
            try:
                download_csv(X, label = "Download the spectral file", fileName = "Spectra", columns = wv)
            except:
                pass
        with col2:
            plotRef(y)
            try:
                download_csv(y, label = "Download the reference value file", fileName = "Reference", columns = ["Reference value"])
            except:
                pass

    elif use_exapmle == "Upload data manually":
        try:
            del st.session_state.X, st.session_state.y, st.session_state.plsModel
        except:
            pass

        st.info("""
                The spectral file you upload needs to meet the requirements
                such that (1) each row is a spectrum of a sample, (2) the first row
                is the wavelength, and (3) the first column is the name of the sample.
                """,
                icon="ℹ️")
        col1, col2 = st.columns([1,1])
        with col1:
            uploaded_file = st.file_uploader("Upload your spectra here","csv")
            if uploaded_file is not None:
                X = pd.read_csv(uploaded_file,index_col=0)
                wv = np.array(X.columns).astype("float")
                sampleNames = X.index
                X = np.array(X)
                plotSPC(X=X,wv=wv)

                st.session_state["X"]=X

        with col2:
            uploaded_file = st.file_uploader("Upload your reference data here","csv")
            if uploaded_file is not None:
                y = np.genfromtxt(uploaded_file, delimiter=',')
                plotRef(y)

                st.session_state["y"]=y


def step2():
    st.markdown("## 2. Cross validtion")
    st.markdown('---')

    X = st.session_state["X"]
    y = st.session_state["y"]

    with st.expander("Set parameters manually for cross validation"):
        n_components = st.slider("The max number of pls components in cross validation.",1,20,10)
        nfold = st.slider("The number of fold in cross validation.",2,20,10)

    plsModel = pls(n_components = n_components)
    plsModel.fit(X,y)
    st.session_state["plsModel"] = plsModel

    yhat = plsModel.predict(X, n_components = np.arange(n_components)+1)
    yhat_cv = plsModel.crossValidation_predict(nfold)
    rmsec = []
    r2 = []
    rmsecv = []
    r2cv = []
    for i in range(yhat_cv.shape[1]):
        report = regresssionReport(y, yhat[:,i])
        reportcv = regresssionReport(y, yhat_cv[:,i])
        rmsec.append(report["rmse"])
        r2.append(report["r2"])
        rmsecv.append(reportcv["rmse"])
        r2cv.append(reportcv["r2"])


    col1, col2 = st.columns([1,1])
    with col1:
        plotRMSECV(rmsec,rmsecv)
    with col2:
        plotR2CV(r2,r2cv)
    st.markdown("### Set the optimal number of component.")
    optLV = st.slider("optLV",1,n_components ,int(np.argmin(rmsecv)+1))
    st.session_state["plsModel"].optLV = optLV

    col1, col2 = st.columns([1,1])
    with col1:
        plotRegressionCoefficients(plsModel.model["B"][2:,optLV-1])
    with col2:
        plotPredictionCV(y, yhat[:,optLV-1], yhat_cv[:,optLV-1])


def step3():
    plsModel = st.session_state["plsModel"]
    st.markdown("## 3. Prediction.")
    st.markdown('---')

    st.markdown("### Import your NIR spectra data for prediction")
    uploaded_file = st.file_uploader("X for prediction",
                                     "csv",label_visibility = "hidden")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file,index_col=0)
        wv = np.array(X.columns).astype("float")
        X = np.array(X)

        col1,col2 = st.columns([1,1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X=X,wv = wv)

        yhat = plsModel.predict(X, n_components = plsModel.optLV)
        with col2:
            st.markdown("### Prediction results")
            st.dataframe(pd.DataFrame(data=yhat,columns=["Prediction"]))
            download_csv(yhat, fileName = "Prediction",
                         columns = ["Prediction"])

        st.markdown("### Import your reference values for validating the prediction")
        uploaded_file = st.file_uploader("y for prediction",
                                         "csv",label_visibility = "hidden")
        if uploaded_file is not None:
            y = np.genfromtxt(uploaded_file, delimiter=',')
            _, col1, _ = st.columns([1,2,1])
            with col1:
                plotPrediction(y,yhat)

st.markdown("""
            # Regression for NIR spectra
            Commonly, the regression of NIR spectra include three steps, (1)Calibration, (2) Cross validation for determine the optimal model parameters, and (3) Prediction on new measured spectra
            """)
step1()
if 'X' in st.session_state and \
    'y' in st.session_state and \
        st.session_state["X"].shape[0] == len(st.session_state["y"]):
    step2()
if 'plsModel' in st.session_state:

    step3()
