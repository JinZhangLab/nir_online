import streamlit as st
import pandas as pd
import numpy as np

from pynir.Calibration import pls
from pynir.CalibrationTransfer import NS_PFCE, SS_PFCE, FS_PFCE, MT_PFCE

from tools.dataManipulationCT import get_Tablet
from tools.dataManipulation import download_csv
from tools.display import plotSPC, plotRef_reg, plotPrediction_reg, plotRegressionCoefficients

allTitle = ["Primary","Second", "Third", "Fourth", "Fifth", "Sixth",
            "Seventh", "Eighth", "Ninth", "Tenth"]

def predict(X, model):
    ones = np.ones((X.to_numpy().shape[0],1))
    X_aug = np.hstack((ones, X.to_numpy()))
    yhat = np.dot(X_aug, np.reshape(model.to_numpy(),(-1,1)))
    return pd.DataFrame(yhat, index=X.index, columns=["Prediction"])

def NS_PFCE_fun(constType="Corr", threshould=0.98):
    ## Obtain standard spectra
    st.header("Data required by NS-PFCE")
    st.info(
        """
        For NS-PFCE, the required inputs are a set of paired standard spectra
        both measured on primary (Xm) and second instruments (Xs). Of note, the
        Xm and Xs shold have  the same number of rows and columns, i.e.,
        the number of samples and variables.
        """
        )

    dataSource = st.radio("Upload data required by NS-PFCE or use our example data.",
                          ["Tablet", "Upload data manually"],
                          horizontal=True,
                          on_change=st.cache_data.clear())

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
            set of 212 samples. The data could be downloaded at [this  Repository](https://github.com/JinZhangLab/PFCE)
            """
            , icon = "⭐")
        data = get_Tablet()
        X1 = data["Trans"]["X"][0]
        X2 = data["Trans"]["X"][1]
        wv = data["wv"].flatten()
        X1 = pd.DataFrame(X1, columns=wv)
        X2 = pd.DataFrame(X2, columns=wv)

    elif dataSource == "Upload data manually":
        cols = st.columns(2)
        with cols[0]:
            uploaded_file1 = st.file_uploader("Upload the standard spectra of Primary instrument","csv")
            if uploaded_file1 is not None:
                X1 = pd.read_csv(uploaded_file1,index_col=0)
                wv = np.array(X1.columns).astype("float")
        with cols[1]:
            uploaded_file2 = st.file_uploader("Upload the standard spectra of Second instrument","csv")
            if uploaded_file2 is not None:
                X2 = pd.read_csv(uploaded_file2,index_col=0)
                wv = np.array(X2.columns).astype("float")

    cols = st.columns(2)
    if "X1" in list(vars().keys()):
        with cols[0]:
            plotSPC(X1, title="NIR spectra of primary instrument")
    if "X2" in list(vars().keys()):
        with cols[1]:
            plotSPC(X2, title="NIR spectra of second instrument")

    ## obtain primary model
    st.header("Primary Model")
    st.info(
        """
        For PFCE, the model of primary instruments are used as
        initial values to estabilshed the model for second instruments. Therefore,
        you need to build a model via PLS or upload an already built model. There is
        an existing primary model for teblet dataset.
        """
        )
    if dataSource == "Tablet":
        n_components = 3
        plsModel = pls(n_components=n_components).fit(data["Cal"]["X"][0], data["Cal"]["y"])
        model = pd.DataFrame(data = plsModel.model['B'][:,-1], index=[-1] + list(wv), columns=["Primary model"])

    elif dataSource == "Upload data manually":
        modelSource = st.radio("build or use your own primary model",
                              ["built primary model", "Upload an estabilished model"],
                              horizontal=True)

        if modelSource == "built primary model":
            cols = st.columns(2)
            with cols[0]:
                uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration","csv")
                if uploaded_file_Xcal is not None:
                    Xcal = pd.read_csv(uploaded_file_Xcal,index_col=0)
                    wv = np.array(Xcal.columns).astype("float")
                    plotSPC(Xcal, title="NIR spectra - Calibration set - Second instrument")

            with cols[1]:
                 uploaded_file_ycal = st.file_uploader("Upload reference value for calibration","csv")
                 if uploaded_file_ycal is not None:
                     ycal = pd.read_csv(uploaded_file_ycal,index_col=0)
                     plotRef_reg(ycal, title = "Reference value - Calibration set")

            if "Xcal" in list(locals().keys()) and "ycal" in list(locals().keys()):
                n_components = st.slider("Number of PLS components",1, min([20, int(np.linalg.matrix_rank(Xcal))]),1)
                plsModel = pls(n_components=n_components).fit(Xcal, ycal)
                model = pd.DataFrame(data = plsModel.model['B'][:,-1], index=[-1] + list(wv), columns=["Primary model"])

        elif modelSource == "Upload an estabilished model":
            uploaded_file_model = st.file_uploader("Upload your model coefficients","csv")
            st.info("""
                    The uploaded model coefficient file needs to be in csv format.
                    You can download a model file from the Regression page in this site to know format requirement.
                    """,
                    icon="ℹ️")
            if uploaded_file_model is not None:
                model = pd.read_csv(uploaded_file_model, index_col=0)

    if "model" in list(vars().keys()):
        _,col1,_ = st.columns([1,2,1])
        with col1:
            plotRegressionCoefficients(model, title = "Model coefficients")


    ## Calibration enhancement
    if "X1" in list(vars().keys()) and "X2" in list(vars().keys()) and "model" in list(vars().keys()):
        startPFCE = st.button("▶ Start calibration enhancement",help="Calibration enhancement with PFCE will take a few nimutes.")
        if startPFCE:
            with st.spinner(
                    """
                    This will take from a few minutes to several minutes depending
                    on the number of variables in NIR spectra.
                    """):
                NS_PFCE_model = NS_PFCE(thres=threshould, constrType=constType).fit(X1.to_numpy(), X2.to_numpy(), model.to_numpy().flatten())

            _,col1,_ = st.columns([1,2,1])
            with col1:
                slaveModel = pd.DataFrame(data = NS_PFCE_model.b2.x, index=[-1] + list(wv), columns=["Slave model"])
                plotRegressionCoefficients(slaveModel, title = "Slave model enhanced by NS-PFCE")

    if "model" in list(vars().keys()) and "NS_PFCE_model" in list(vars().keys()):
        if dataSource == "Tablet":
            Xtest1 = data["Test"]["X"][0]
            Xtest2 = data["Test"]["X"][1]
            ytest = data["Test"]["y"]
            Xtest1 = pd.DataFrame(Xtest1, columns=wv)
            Xtest2 = pd.DataFrame(Xtest2, columns=wv)
            ytest = pd.DataFrame(ytest, columns=["Reference"])

        elif dataSource == "Upload data manually":

            cols = st.columns(3)
            with cols[0]:
                uploaded_file_Xtest1 = st.file_uploader("Upload spectra of primary instrument for prediction","csv")
                if uploaded_file_Xtest1 is not None:
                    Xtest1 = pd.read_csv(uploaded_file_Xtest1,index_col=0)
                    wv = np.array(Xtest1.columns).astype("float")
                    plotSPC(Xtest1, title="Prediction set of Primary instruments")

            with cols[1]:
                uploaded_file_Xtest2 = st.file_uploader("Upload spectra of second instrument for prediction","csv")
                if uploaded_file_Xtest2 is not None:
                    Xtest2 = pd.read_csv(uploaded_file_Xtest2,index_col=0)
                    wv = np.array(Xtest2.columns).astype("float")
                    plotSPC(Xtest2, title="Prediction set of Second instruments")

            with cols[2]:
                 uploaded_file_ytest = st.file_uploader("Upload reference value for validating prediction","csv")
                 if uploaded_file_ytest is not None:
                     ytest = pd.read_csv(uploaded_file_ytest,index_col=0)
                     plotRef_reg(ytest)

    if "Xtest1" in list(vars().keys()) and \
        "Xtest2" in list(vars().keys()) and \
        "ytest" in list(vars().keys()) and \
        "model" in list(vars().keys()):
        ## predict the prediction set of both instrument with primary model before calibation enhancement
        st.header("Prediction before calibration enhancement")
        st.info(
            """
            In general, the primary model predicts relatively well for the primary instrument,
            while significantly lower for the second machine. For validting the calibration enhancement
            effect, you need upload the spectra in prediction set of primary and second instruments,
            as well as the corresponding reference values. These data has been integraed into the plateform
            for example dataset.
            """
            )

        st.markdown("Predictions from applying the primary model ***directly*** to the prediction set")
        cols = st.columns(2)
        with cols[0]:
            yhatMaster = pd.DataFrame(data = [ytest.to_numpy().flatten(), predict(Xtest1, model).to_numpy().flatten()],
                                         index=["Reference", "Master"], columns=ytest.index)
            plotPrediction_reg(yhatMaster,
                           xlabel="Reference", ylabel="Prediction",
                           title="Prediction of master spectra before calibration enhancement")
        with cols[1]:
            yhatSlave = pd.DataFrame(data = [ytest.to_numpy().flatten(), predict(Xtest2, model).to_numpy().flatten()],
                                      index=["Reference", "Slave"], columns=ytest.index)           
            plotPrediction_reg(yhatSlave,
                           xlabel="Reference", ylabel="Prediction",
                           title="Prediction of slave spectra before calibration enhancement")

    ## predict the prediction set of both instrument with primary model After calibation enhancement
    if "Xtest2" in list(vars().keys()) and \
        "ytest" in list(vars().keys()) and \
        "NS_PFCE_model" in list(vars().keys()):
        yhat2_NS_PFCE = NS_PFCE_model.transform(Xtest2.to_numpy())
        st.header("Prediction before calibration enhancement")
        st.markdown(
            """Predictions of the prediction set of secondard instrument by applying the primary model ***After*** calibration
            enhancement
            """)
        _,col1,_ = st.columns([1,2,1])
        with col1:
            yhatWithCT = pd.DataFrame(data = [ytest.to_numpy().flatten(), yhat2_NS_PFCE.flatten()],
                                index=["Reference", "Slave"], columns=ytest.index)
            plotPrediction_reg(yhatWithCT,
                           title="Prediction of slave After calibration enhanced by NS-PFCE")

# Page content
st.set_page_config(page_title="NIR Online-Calibration Enhancement", page_icon=":rocket:", layout="wide")

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
