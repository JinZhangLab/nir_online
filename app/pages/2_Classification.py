import streamlit as st

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from pynir.Calibration import plsda, binaryClassificationReport, multiClassificationReport
from pynir.utils import simulateNIR

from tools.display import (plotSPC, plotRef_clf,
                           plotAccuracyCV, plot_confusion_matrix)

from tools.dataManipulation import download_csv


def step1():
    st.markdown("# Step 1. Load data")
    st.markdown("---")

    use_exapmle = st.radio("Upload your data or use our example.", [
                           "Example data 1", "Upload data manually"])

    if use_exapmle == "Example data 1":
        try:
            del st.session_state.X, st.session_state.y, st.session_state.plsdaModel
        except:
            pass
        X, y, wv = simulateNIR(refType=3)
        sampleNames = [f"Sample_{i}" for i in range(X.shape[0])]
        X = pd.DataFrame(X, columns=wv, index=sampleNames)
        y = pd.DataFrame(y, columns=["Reference values"], index=sampleNames)

        st.session_state["X"] = X
        st.session_state["y"] = y

        col1, col2 = st.columns([1, 1])
        with col1:
            plotSPC(X)

        with col2:
            plotRef_clf(y)

    elif use_exapmle == "Upload data manually":
        try:
            del st.session_state.X, st.session_state.y, st.session_state.plsdaModel
        except:
            pass

        st.info("""
                The spectral file you upload needs to meet the requirements
                such that (1) each row is a spectrum of a sample, (2) the first row
                is the wavelength, and (3) the first column is the name of the sample.
                """,
                icon="ℹ️")
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
                st.markdown("### Reference values uploaded.")
                
                plotRef_clf(y)
                st.session_state["y"] = y


def step2():
    st.markdown("# Step 2. Cross validtion")
    st.markdown("---")

    X = st.session_state["X"]
    y = st.session_state["y"]

    with st.container():
        st.markdown("### Set parameters manually for cross validation")
        n_components = st.slider(
            "The max number of component in pls calibration.", 1, 20, 10)
        nfold = st.slider("The number of fold in cross validation.", 2, 20, 10)

    plsdaModel = plsda(n_components=n_components)
    plsdaModel.fit(X.to_numpy(), y.to_numpy())
    st.session_state["plsdaModel"] = plsdaModel

    yhat_cv = plsdaModel.crossValidation_predict(nfold)
    accuracy_cv = []
    f1_cv = []
    for i in range(yhat_cv.shape[1]):
        if len(plsdaModel.lda.classes_) == 2:
            report_cv = binaryClassificationReport(y, yhat_cv[:, i])
            accuracy_cv.append(report_cv["accuracy"])
            f1_cv.append(report_cv["f1"])
        elif len(plsdaModel.lda.classes_) > 2:
            report_cv = multiClassificationReport(y, yhat_cv[:, i])
            accuracy_cv.append(np.mean([rep["accuracy"]
                               for rep in report_cv.values()]))
            f1_cv.append(np.mean([rep["f1"] for rep in report_cv.values()]))

    col1, col2 = st.columns([1, 1])
    with col1:
        plotAccuracyCV(accuracy_cv, labels="accuracy")
    with col2:
        plotAccuracyCV(f1_cv, labels="f1")
    st.markdown("### Set the optimal number of component.")
    optLV = st.slider("optLV", 1, n_components, int(np.argmax(accuracy_cv)+1))

    plsdaModel = plsda(n_components=optLV)
    plsdaModel.fit(X, y)
    st.session_state["plsdaModel"] = plsdaModel

    col1, col2 = st.columns([1, 1])
    with col1:
        yhat_c = plsdaModel.predict(X)
        cm_c = confusion_matrix(y, yhat_c)
        cm_c = pd.DataFrame(data=cm_c, index=plsdaModel.lda.classes_, columns=plsdaModel.lda.classes_)
        plot_confusion_matrix(cm_c, plsdaModel.lda.classes_, 
                              title="Confusion Matrix-training set")
    with col2:
        cm_cv = confusion_matrix(y, yhat_cv[:, optLV-1])
        cm_cv = pd.DataFrame(data=cm_cv, index=plsdaModel.lda.classes_, columns=plsdaModel.lda.classes_)
        plot_confusion_matrix(cm_cv, plsdaModel.lda.classes_, 
                              title="Confusion Matrix-Cross validation")


def step3():
    st.markdown("# Step 3. Prediction")
    st.markdown("---")
    plsdaModel = st.session_state["plsdaModel"]

    st.markdown("Import your NIR spectra data here for prediction")
    uploaded_file = st.file_uploader("X for prediction",
                                     "csv", label_visibility="hidden")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        wv = np.array(X.columns).astype("float")
        sampleName = X.index

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### NIR spectra for prediction")
            plotSPC(X)

        yhat = plsdaModel.predict(X.to_numpy())
        yhat = pd.DataFrame(data=yhat,index = sampleName, columns=["Prediction"])
        with col2:
            st.markdown("### Prediction results")
            st.dataframe(yhat)
            download_csv(yhat, fileName="Prediction", label="Download results",
                         columns=["Prediction"])
        st.markdown(
            "### Import your reference values for validatation")
        uploaded_file = st.file_uploader("y for prediction",
                                         "csv", label_visibility="hidden")
        if uploaded_file is not None:
            y = pd.read_csv(uploaded_file, index_col=0)
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                cm = plsdaModel.get_confusion_matrix(X.to_numpy(), y.to_numpy())
                cm = pd.DataFrame(data=cm, index=plsdaModel.lda.classes_, columns=plsdaModel.lda.classes_)
                plot_confusion_matrix(cm, plsdaModel.lda.classes_,
                                      title="Confusion Matrix-Prediction")

# page content


st.set_page_config(page_title="NIR Online-Classification",
                   page_icon="📈", layout="wide")

step1()

if 'X' in st.session_state and \
    'y' in st.session_state and \
        st.session_state["X"].shape[0] == len(st.session_state["y"]):
    step2()
if 'plsdaModel' in st.session_state:
    step3()
