import streamlit as st

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from pynir.Calibration import plsda, binaryClassificationReport,multiClassificationReport

from display import (plotSPC, plotRef_clf, 
                     plotAccuracyCV, plot_confusion_matrix,
                     plotPredictionCV,plotPrediction, 
                     plotRegressionCoefficients, download_csv)

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
            plotRef_clf(y)
            st.session_state["y"]=y 

def step2():
    X = st.session_state["X"]
    y = st.session_state["y"]
    st.markdown("# 2. Cross validtion")
    with st.expander("Set parameters manually for cross validation"):
        ncomp = st.slider("The max number of component in pls calibration.",1,20,10)
        nfold = st.slider("The number of fold in cross validation.",2,20,10)
    
    plsdaModel = plsda(nlv = ncomp)
    plsdaModel.fit(X,y)
    st.session_state["plsdaModel"] = plsdaModel
    

    yhat_cv = plsdaModel.crossValidation_predict(nfold)
    accuracy_cv = []
    f1_cv = []
    for i in range(yhat_cv.shape[1]):
        if len(plsdaModel.classes) == 2:
            report_cv = binaryClassificationReport(y, yhat_cv[:,i])
            accuracy_cv.append(report_cv["accuracy"])
            f1_cv.append(report_cv["f1"])
        elif len(plsdaModel.classes) > 2:
            report_cv = multiClassificationReport(y, yhat_cv[:,i])
            accuracy_cv.append(np.mean([rep["accuracy"] for rep in report_cv.values()]))
            f1_cv.append(np.mean([rep["f1"] for rep in report_cv.values()]))
    
    
    col1, col2 = st.columns([1,1])
    with col1:
        plotAccuracyCV(accuracy_cv,labels = "accuracy")
    with col2:
        plotAccuracyCV(f1_cv,labels = "f1")
    st.markdown("### Set the optimal number of component.")
    optLV = st.slider("optLV",1,ncomp ,int(np.argmax(accuracy_cv)+1))
    

    plsdaModel = plsda(nlv = optLV)
    plsdaModel.fit(X,y)
    st.session_state["plsdaModel"] = plsdaModel
     
    
    col1, col2 = st.columns([1,1])
    with col1:
        yhat_c = plsdaModel.predict(X)
        cm_c = confusion_matrix(y,yhat_c)
        plot_confusion_matrix(cm_c,np.unique(y),normalize=False,
                              title="Prediction on training set")
    with col2:
        cm_cv = confusion_matrix(y, yhat_cv[:,optLV-1])
        plot_confusion_matrix(cm_cv, np.unique(y),normalize=False,
                              title="Cross validation")
    

def step3():
    plsdaModel = st.session_state["plsdaModel"]

    st.markdown("# 3. Validation")
    
    st.markdown("### Import your NIR spectra data for prediction")
    uploaded_file = st.file_uploader("X for prediction",
                                     "csv",label_visibility = "hidden")
    if uploaded_file is not None:
        X = np.genfromtxt(uploaded_file, delimiter=',')     
        col1,col2 = st.columns([1,1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X=X)
        
        yhat = plsdaModel.predict(X)
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
                cm = plsdaModel.get_confusion_matrix(X,y)
                plot_confusion_matrix(cm, np.unique(y),normalize=False,
                                      title="Prediction on uploaded spectra")
    
step1()
st.markdown('---')
if 'X' in st.session_state and \
    'y' in st.session_state and \
        st.session_state["X"].shape[0] == len(st.session_state["y"]):
    step2()
if 'plsdaModel' in st.session_state:
    st.markdown('---')
    step3()
    