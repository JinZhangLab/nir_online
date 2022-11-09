import streamlit as st
import pandas as pd
import numpy as np

from pynir.Calibration import pls
from pynir.CalibrationTransfer import NS_PFCE, SS_PFCE, FS_PFCE, MT_PFCE

from tools.dataManipulationCT import get_Tablet
from tools.display import plotSPC, plotRef, plotPrediction

allTitle = ["Primary","Second", "Third", "Fourth", "Fifth", "Sixth", 
            "Seventh", "Eighth", "Ninth", "Tenth"]

def predict(X, model):
    ones = np.ones((X.shape[0],1))
    X_aug = np.hstack((ones, X))
    yhat = np.dot(X_aug, np.reshape(model,(-1,1)))
    return yhat
    
def NS_PFCE_fun(constType="Corr", threshould=0.98):
    dataSource = st.radio("Upload your data or use our example.",
                          ["Tablet", "Upload data manually"],
                          horizontal=True)
    
    ## Obtain standard spectra
    if dataSource == "Tablet":
        st.info(
            """
            The example dataset consist of NIR spectra of 655 pharmaceutical tablets
            measured on two NIR instruments in the wavelength range of 600-1898 nm with 
            the digital interval of 2 nm. The noise region of 1794-1898 nm, and 13
            outlier samples were removed as suggested [see PFCE article]. For each
            template sample, active pharmaceutical ingredient (API) has been measured
            as the target of calibration. The remained 642 samples were divided into
            a calibration set of 400 samples, standard set of 30 samples and prediction
            set of 212 samples.
            """
            , icon = "⭐")
        data = get_Tablet()
        X1 = data["Trans"]["X"][0]
        X2 = data["Trans"]["X"][1]
        wv = data["wv"]
    elif dataSource == "Upload data manually":
        st.info(
            """
            For NS-PFCE, the required inputs are a set of paired standard spectra
            both measured on primary (Xm) and second instruments (Xs). Of note, the
            Xm and Xs shold have  the same number of rows and columns, i.e., 
            the number of samples and variables.
            """
            )
        cols = st.columns(2)
        with cols[0]:
            uploaded_file1 = st.file_uploader("Upload the standard spectra of Primary instrument","csv")
            if uploaded_file1 is not None:
                X1 = pd.read_csv(uploaded_file1,index_col=0)
                wv = np.array(X1.columns).astype("float")
                X1 = np.array(X1)      
        with cols[1]:
            uploaded_file2 = st.file_uploader("Upload the standard spectra of Second instrument","csv")
            if uploaded_file2 is not None:
                X2 = pd.read_csv(uploaded_file2,index_col=0)
                wv = np.array(X2.columns).astype("float")
                X2 = np.array(X2)
                
    cols = st.columns(2)
    if "X1" in list(vars().keys()):
        with cols[0]:
            plotSPC(X1, wv = wv, title="Primary")
    if "X2" in list(vars().keys()):
        with cols[1]:
            plotSPC(X2, wv = wv, title="Second")

    ## obtain primary model
    st.info(
        """
        For the PFCE, the model estabiliesh for primary instruments are used as 
        initional values to estabilshed the model for second instruments. Therefore,
        you need to build a model via PLS or upload an already built model. There is
        an existing primary model for teblet dataset.
        """
        )
    if dataSource == "Tablet":

        nComp = 3
        wv = data["wv"]
        plsModel = pls(nComp=nComp).fit(data["Cal"]["X"][0], data["Cal"]["y"])
        model = plsModel.model['B'][:,-1]

    elif dataSource == "Upload data manually":
        modelSource = st.radio("build or use your own primary model",
                              ["built primary model", "Upload own model"],
                              horizontal=True)
        
        if modelSource == "built primary model":
            cols = st.columns(2)
            with cols[0]:
                uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration","csv")
                if uploaded_file_Xcal is not None:
                    Xcal = pd.read_csv(uploaded_file_Xcal,index_col=0)
                    wv = np.array(Xcal.columns).astype("float")
                    Xcal = np.array(Xcal)
                    plotSPC(Xcal, wv = wv, title="Second")
             
            with cols[1]:
                 uploaded_file_ycal = st.file_uploader("Upload reference value for calibration","csv")
                 if uploaded_file_ycal is not None:
                     ycal = pd.read_csv(uploaded_file_ycal,index_col=0)
                     ycal = np.array(ycal)
                     plotRef(ycal)
            
            if "Xcal" in list(locals().keys()) and "ycal" in list(locals().keys()):
                nComp = st.slider("Number of PLS components",1, min([20, int(np.linalg.matrix_rank(Xcal))]),1)
                plsModel = pls(nComp=nComp).fit(Xcal, ycal)
                model = plsModel.model['B'][:,-1]
                
        elif modelSource == "Upload own model":
            uploaded_file_model = st.file_uploader("Upload your model coefficients","csv")
            st.info("""
                    The uploaded model coefficient file needs to be in csv format, 
                    with n+1 rows and 1 column, where the first number is the intercept
                    term and the following n numbers are the coefficient terms.
                    """,
                    icon="ℹ️")
            if uploaded_file_model is not None:
                model = np.loadtxt(uploaded_file_model, delimiter=',')

    if "model" in list(vars().keys()):
        _,col1,_ = st.columns([1,2,1])
        with col1:
            plotSPC(model[1:], wv =wv, title = "Model coefficients")
    
    ## predict the prediction set of both instrument with primary model before calibation enhancement
    st.info(
        """
        In general, the primary model predicts relatively well for the primary instrument,
        while significantly lower for the second machine. For validting the calibration enhancement
        effect, you need upload the spectra in prediction set of primary and second instruments, 
        as well as the corresponding reference values. These data has been integraed into the plateform
        for example dataset.
        """
        )
    if "model" in list(vars().keys()):
        if dataSource == "Tablet":
            Xtest1 = data["Test"]["X"][0]
            Xtest2 = data["Test"]["X"][1]
            ytest = data["Test"]["y"]
        elif dataSource == "Upload data manually":
            cols = st.columns(3)
            with cols[0]:
                uploaded_file_Xtest1 = st.file_uploader("Upload spectra of primary instrument for prediction","csv")
                if uploaded_file_Xtest1 is not None:
                    Xtest1 = pd.read_csv(uploaded_file_Xtest1,index_col=0)
                    wv = np.array(Xtest1.columns).astype("float")
                    Xtest1 = np.array(Xtest1)
                    plotSPC(Xtest1, wv = wv, title="Prediction set of Primary instruments")

            with cols[1]:
                uploaded_file_Xtest2 = st.file_uploader("Upload spectra of second instrument for prediction","csv")
                if uploaded_file_Xtest2 is not None:
                    Xtest2 = pd.read_csv(uploaded_file_Xtest2,index_col=0)
                    wv = np.array(Xtest2.columns).astype("float")
                    Xtest2 = np.array(Xtest2)
                    plotSPC(Xtest2, wv = wv, title="Prediction set of Second instruments")

            with cols[2]:
                 uploaded_file_ytest = st.file_uploader("Upload reference value for validating prediction","csv")
                 if uploaded_file_ytest is not None:
                     ytest = pd.read_csv(uploaded_file_ytest,index_col=0)
                     ytest = np.array(ytest)
                     plotRef(ytest)
    
    st.markdown("Predictions from applying the primary model ***directly*** to the prediction set")    
    if "Xtest1" in list(vars().keys()) and \
        "Xtest2" in list(vars().keys()) and \
        "ytest" in list(vars().keys()) and \
        "model" in list(vars().keys()):
        cols = st.columns(2)
        with cols[0]:
            plotPrediction(ytest, predict(Xtest1,model), 
                           title="Outcomes of prediction set of " + allTitle[0] + " instrument with primary model Before enhanced")
        with cols[1]:
            plotPrediction(ytest, predict(Xtest1,model), 
                           title="Outcomes of prediction set of " + allTitle[1] + " instrument  with primary model Before enhanced")

    ## Calibration enhancement
    if "X1" in list(vars().keys()) and "X2" in list(vars().keys()) and "model" in list(vars().keys()):
        with st.spinner(
                """
                Wait for the calulcating of NS-PFCE..., 
                this will take from a few minutes to several minutes depending
                on the number of variables you calculate for the NIR spectra.
                To reduce computation time, we recommend selecting representative 
                variables before calibration enhancement.
                """):
            NS_PFCE_model = NS_PFCE(thres=threshould, constrType=constType).fit(X1,X2,model)
        
        _,col1,_ = st.columns([1,2,1])
        with col1:
            plotSPC(np.ravel(NS_PFCE_model.b2.x)[1:], wv =wv, title = "Model coefficients of second instrument enhanced by NS-PFCE")

    ## predict the prediction set of both instrument with primary model After calibation enhancement
    if "Xtest2" in list(vars().keys()) and \
        "ytest" in list(vars().keys()) and \
        "NS_PFCE_model" in list(vars().keys()):
        yhat2_NS_PFCE = NS_PFCE_model.transform(Xtest2)
        st.markdown("Predictions from applying the primary model ***After*** calibration enhancement to the prediction set of secondard instrument")
        _,col1,_ = st.columns([1,2,1])
        with col1:
            plotPrediction(ytest, yhat2_NS_PFCE, 
                           title="Prediction of " + allTitle[1] + " instrument After calibration enhanced by NS-PFCE")

# Main Page
st.markdown("# Calibration Transfer/Enhancement")


st.info(
    """
    Parameter free calibration enhancement (PFCE)  is a formal unified NIR 
    spectral model enhancement framework proposed by our team that can cope
    with many different known conditions without complex hyperparameter 
    optimization. The framework includes four main algorithms, nonsupervised(NS-), 
    semisupervised(SS-) , fullsupervised(FS-) and multitask(MT-) PFCE. For more
    information, please refer to this [Article](https://www.sciencedirect.com/science/article/abs/pii/S0003267020311107).
    """
    )

method = st.radio("Select a method", 
                  ["NS-PFCE","SS-PFCE","FS-PFCE","MT-PFCE"],
                  horizontal=True)
constType = st.radio("Constraint Type", ["Corr", "L2", "L1"],horizontal=True)

threshould = st.slider("Constraint threshould", 0.00, 1.00, 0.98)

ConstMap = {"Corr":1, "L2":2, "L1":3}
if method == "NS-PFCE":
    NS_PFCE_fun(constType=ConstMap[constType], threshould=threshould)

elif method == "SS-PFCE":
    pass
elif method == "FS-PFCE":
    pass
elif method == "MT-PFCE":
    pass
  