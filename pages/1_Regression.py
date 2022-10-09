import streamlit as st

import numpy as np
import pandas as pd

from pynir.Calibration import pls, regresssionReport

from display import (plotSPC, plotRef, plotRMSECV, plotR2CV, 
                     plotPredictionCV,plotPrediction, plotRegressionCoefficients)

def step1():
    st.markdown("# 1. Load data")
        
    for key in st.session_state.keys():
        del st.session_state[key]
    st.markdown("### Import your NIR spectra and reference data.")
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_file = st.file_uploader("Upload your spectra here","csv")
        if uploaded_file is not None:
            X = np.genfromtxt(uploaded_file, delimiter=',')        
    
            st.markdown("### NIR spectra uploaded.")
            plotSPC(X=X)
            st.session_state["X"]=X
    
    with col2:    
        uploaded_file = st.file_uploader("Upload your reference data here","csv")
        if uploaded_file is not None:
            y = np.genfromtxt(uploaded_file, delimiter=',')        
            st.markdown("### Reference values uploaded.")
            plotRef(y)
            st.session_state["y"]=y 

def step2():
    X = st.session_state["X"]
    y = st.session_state["y"]
    st.markdown("# 2. Cross validtion for selecting optimal number of PLS components.")
    with st.expander("Set parameters manually for cross validation"):
        ncomp = st.slider("The max number of component in pls calibration.",1,20,10)
        nfold = st.slider("The number of fold in cross validation.",2,20,10)
    
    plsModel = pls(nlv = ncomp)
    plsModel.fit(X,y)
    st.session_state["plsModel"] = plsModel
    
    yhat = plsModel.predict(X)
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
    optLV = st.slider("optLV",1,ncomp ,int(np.argmin(rmsecv)+1))
    st.session_state["plsModel"].optLV = optLV
    st.session_state["plsModel"].optLV
    
    col1, col2 = st.columns([1,1])
    with col1:
        plotRegressionCoefficients(plsModel.model["B"][2:,optLV-1])
    with col2:
        plotPredictionCV(y, yhat[:,optLV-1], yhat_cv[:,optLV-1])
    

def step3():
    plsModel = st.session_state["plsModel"]
    plsModel.optLV

    st.markdown("# 3. Validation of the estabilished model.")
    
    st.markdown("### Import your NIR spectra data for prediction")
    uploaded_file = st.file_uploader("X for prediction",
                                     "csv",label_visibility = "hidden")
    if uploaded_file is not None:
        X = np.genfromtxt(uploaded_file, delimiter=',')     
        col1,col2 = st.columns([1,1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X=X)
        
        yhat = plsModel.optPredict(X)
        with col2:
            st.markdown("### Prediction results")
            st.dataframe(pd.DataFrame(data=yhat,columns=["Prediction"]))
            
        st.markdown("### Import your reference values for validating the prediction")      
        uploaded_file = st.file_uploader("y for prediction",
                                         "csv",label_visibility = "hidden")
        if uploaded_file is not None:
            y = np.genfromtxt(uploaded_file, delimiter=',')
            _, col1, _ = st.columns([1,2,1])
            with col1:
                plotPrediction(y,yhat)
    
step1()
st.markdown('---')
if 'X' in st.session_state and \
    'y' in st.session_state and \
        st.session_state["X"].shape[0] == len(st.session_state["y"]):
    step2()
if 'plsModel' in st.session_state:
    st.markdown('---')
    step3()
    