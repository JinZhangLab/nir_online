import streamlit as st
import pandas as pd
import numpy as np

from pynir.Calibration import pls
from pynir.CalibrationTransfer import NS_PFCE, SS_PFCE, FS_PFCE, MT_PFCE

from tools.dataManipulation import get_Tablet, predict_reg
from tools.dataManipulation import download_csv
from tools.display import plotSPC, plotRef_reg, plotPrediction_reg, plotRegressionCoefficients

allTitle = ["Primary", "Second", "Third", "Fourth", "Fifth", "Sixth",
            "Seventh", "Eighth", "Ninth", "Tenth"]

def changeCT_state():
    st.session_state.ct += 1


def NS_PFCE_fun(constType="Corr", threshould=0.98):
    # Obtain the data required by NS-PFCE
    st.header("Data required by NS-PFCE")
    st.info(
        """
        For NS-PFCE, the required inputs are a set of paired standard spectra
        both measured on primary (Xm) and second instruments (Xs). Of note, the
        Xm and Xs shold have  the same number of rows and columns, i.e., the
        number of samples and variables.
        """
    )

    dataSource = st.radio("Upload data required by NS-PFCE or use our example data.",
                          ["Tablet", "Upload data manually"],
                          horizontal=True,
                          on_change=changeCT_state,
                          key="calibration_transfer_NS_PFCE_data_selection")

    if dataSource == "Tablet":
        st.info(
            """
            The example dataset provided for calibration transfer consists of
            NIR spectra of 655 pharmaceutical tablets measured on two NIR
            instruments in the wavelength range of 600-1898 nm with a digital
            interval of 2 nm. The noise region of 1794-1898 nm and 13 outlier
            samples were removed as suggested in the [PFCE
            article](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637?via%3Dihub).

            For each template sample, the active pharmaceutical ingredient (API)
            has been measured as the target of calibration. The remaining 642
            samples were divided into a calibration set of 400 samples, a
            standard set of 30 samples, and a prediction set of 212 samples.

            
            To download the example dataset, please use the "example dataset"
            function on the utils page of this site. The dataset has already
            been preprocessed and divided into the calibration, standard, and
            prediction sets as described above. We hope that this dataset will
            be useful to you in your research, and we encourage you to cite the
            relevant literature if you use it in your work. 
            """)
        data = get_Tablet()
        X1 = data["Trans"]["X"][0]
        X2 = data["Trans"]["X"][1]
        wv = data["wv"].flatten()
        X1 = pd.DataFrame(X1, columns=wv)
        X2 = pd.DataFrame(X2, columns=wv)

    elif dataSource == "Upload data manually":
        cols = st.columns(2)
        with cols[0]:
            uploaded_file1 = st.file_uploader(
                "Upload the standard spectra of Primary instrument", "csv", 
                key="CT_NS_PFCE_Xstd_1"+str(st.session_state.ct))
            if uploaded_file1 is not None:
                X1 = pd.read_csv(uploaded_file1, index_col=0)
                wv = np.array(X1.columns).astype("float")
        with cols[1]:
            uploaded_file2 = st.file_uploader(
                "Upload the standard spectra of Second instrument", "csv", 
                key="CT_NS_PFCE_Xstd_2"+str(st.session_state.ct))
            if uploaded_file2 is not None:
                X2 = pd.read_csv(uploaded_file2, index_col=0)
                wv = np.array(X2.columns).astype("float")

    cols = st.columns(2)
    if "X1" in locals():
        with cols[0]:
            plotSPC(X1, title="NIR spectra of standard samples from the primary/master instrument")
    if "X2" in locals():
        with cols[1]:
            plotSPC(X2, title="NIR spectra of standard samples from the second/slave instrument")

    # Obtain primary model
    if "X1" in locals() and "X2" in locals():
        st.header("Primary Model")
        st.info(
            """
            For PFCE, the model of primary instruments are used as initial
            values to estabilshed the model for second instruments. Therefore,
            you need to build a model via PLS or upload an already built model.
            There is an existing primary model for teblet dataset.
            """
        )
        if dataSource == "Tablet":
            n_components = 3
            plsModel = pls(n_components=n_components).fit(data["Cal"]["X"][0], data["Cal"]["y"])
            model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1, -1), columns=[-1] + list(wv), index=["Primary model"])

        elif dataSource == "Upload data manually":
            modelSource = st.radio("build or use your own primary model",
                                ["built primary model",
                                    "Upload an estabilished model"],
                                horizontal=True,
                                key="CT_NS_PFCE_primary_model_selection")

            if modelSource == "built primary model":
                cols = st.columns(2)
                with cols[0]:
                    uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration", "csv",
                                                        key="CT_NS_PFCE_Xcal_1"+str(st.session_state.ct))
                    if uploaded_file_Xcal is not None:
                        Xcal = pd.read_csv(uploaded_file_Xcal, index_col=0)
                        wv = np.array(Xcal.columns).astype("float")
                        plotSPC(Xcal, title="NIR spectra - Calibration set - Second instrument")

                with cols[1]:
                    uploaded_file_ycal = st.file_uploader("Upload reference value for calibration", "csv",
                                                        key="CT_NS_PFCE_ycal"+str(st.session_state.ct))
                    if uploaded_file_ycal is not None:
                        ycal = pd.read_csv(uploaded_file_ycal, index_col=0)
                        plotRef_reg(
                            ycal, title="Reference value - Calibration set")

                if "Xcal" in locals() and "ycal" in locals():
                    n_components = st.slider("Number of PLS components", 
                                            1, min([20, int(np.linalg.matrix_rank(Xcal))]), 1)
                    plsModel = pls(n_components=n_components).fit(Xcal, ycal)
                    model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1, -1), columns=[-1] + list(wv), 
                                        index =["Primary model"])

            elif modelSource == "Upload an estabilished model":
                uploaded_file_model = st.file_uploader("Upload your model coefficients", "csv",
                                                    key="CT_NS_PFCE_primary_model"+str(st.session_state.ct))
                st.info("""
                        The uploaded model coefficient file needs to be in csv
                        format. You can download a model file from the
                        Regression page in this site to know format requirement.
                        """)
                if uploaded_file_model is not None:
                    model = pd.read_csv(uploaded_file_model, index_col=0)

        if "model" in locals():
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                plotRegressionCoefficients(model, title="Primary Model")

    # Calibration enhancement with NS-PFCE
    if "X1" in locals() and "X2" in locals() and "model" in locals():
        st.header("Slave Model enhanced by NS-PFCE")
        st.info(
            """
            The enhanced model can be applied to predict the spectra with  the
            widget of **Predict with regression model** in the **Utils** page of
            this site.
            """
        )
        with st.spinner(
                """
                Calibration enhancement in progress...This will take from a few
                minutes to several minutes depending on the number of variables
                in NIR spectra.
                """):
            NS_PFCE_model = NS_PFCE(thres=threshould, constrType=constType).fit(
                X1.to_numpy(), X2.to_numpy(), model.to_numpy().flatten())

        _, col1, _ = st.columns([1, 2, 1])
        with col1:
            slaveModel = pd.DataFrame(data=np.reshape(NS_PFCE_model.b2.x, (1, -1)), 
                                        columns=[-1] + list(wv), index=["Slave model"])
            plotRegressionCoefficients(slaveModel, title="Slave model enhanced by NS-PFCE")


def SS_PFCE_fun(constType="Corr", threshould=0.98):
    # Obtain the data required by SS-PFCE
    st.header("Data required by SS-PFCE")
    st.info(
        """
        The SS-PFCE method requires a set of spectra measured only on the
        second/slave instruments (Xs) and the corresponding reference values (y)
        as inputs. It is important to note that the format of Xs and y should
        comply with the format requirements of this platform. Specifically, the
        samples should be arranged in rows in both the spectral and reference
        value files, with the first column and row representing the sample name
        and wavelength/characteristics of "Reference value", respectively.
        """
    )

    dataSource = st.radio("Upload data required by SS-PFCE or use our example data.",
                        ["Tablet", "Upload data manually"], horizontal=True,
                        on_change=changeCT_state,  key="calibration_transfer_SS_PFCE_data_selection")
    
    if dataSource == "Tablet":
        st.info(
            """
            The example dataset provided for calibration transfer consists of
            NIR spectra of 655 pharmaceutical tablets measured on two NIR
            instruments in the wavelength range of 600-1898 nm with a digital
            interval of 2 nm. The noise region of 1794-1898 nm and 13 outlier
            samples were removed as suggested in the [PFCE
            article](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637?via%3Dihub).

            For each template sample, the active pharmaceutical ingredient (API)
            has been measured as the target of calibration. The remaining 642
            samples were divided into a calibration set of 400 samples, a
            standard set of 30 samples, and a prediction set of 212 samples.

            
            To download the example dataset, please use the "example dataset"
            function on the utils page of this site. The dataset has already
            been preprocessed and divided into the calibration, standard, and
            prediction sets as described above. We hope that this dataset will
            be useful to you in your research, and we encourage you to cite the
            relevant literature if you use it in your work. 
            """)
        data = get_Tablet()
        X2 = data["Trans"]["X"][1]
        y = data["Trans"]["y"]
        wv = data["wv"].flatten()
        X2 = pd.DataFrame(X2, columns=wv)
        y = pd.DataFrame(y, columns=["Reference"])

    elif dataSource == "Upload data manually":
        cols = st.columns(2)
        with cols[0]:
            uploaded_file1 = st.file_uploader(
                "Upload the standard spectra from second/slave instrument", "csv",
                key="CT_SS_PFCE_Xstd_2"+str(st.session_state.ct))
            if uploaded_file1 is not None:
                X2 = pd.read_csv(uploaded_file1, index_col=0)
                wv = np.array(X2.columns).astype("float")
        with cols[1]:
            uploaded_file2 = st.file_uploader(
                "Upload the reference values for the standard", "csv",
                key="CT_SS_PFCE_ystd"+str(st.session_state.ct))
            if uploaded_file2 is not None:
                y = pd.read_csv(uploaded_file2, index_col=0)

    cols = st.columns(2)
    if "X2" in locals():
        with cols[0]:
            plotSPC(X2, title="NIR spectra of strandard samples from the second/slave instrument")
    if "y" in locals():
        with cols[1]:
            plotRef_reg(y, title="Reference values of standard samples")

    # Obtain the primary/master model
    if "X2" in locals() and "y" in locals():
        st.header("Primary/master model")
        st.info(
            """
            For PFCE, the model of primary instruments are used as initial
            values to estabilshed the model for second instruments. Therefore,
            you need to build a model via PLS or upload an already built model.
            There is an existing primary model for teblet dataset.
            """)
        if dataSource == "Tablet":
                n_components = 3
                plsModel = pls(n_components=n_components).fit(data["Cal"]["X"][0], data["Cal"]["y"])
                model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1), columns=[-1] + list(wv), index=["Primary model"])

        elif dataSource == "Upload data manually":
            modelSource = st.radio("build or use your own primary model",
                                ["built primary model",
                                    "Upload an estabilished model"],
                                horizontal=True, key="CT_SS_PFCE_primary_model_selection")

            if modelSource == "built primary model":
                cols = st.columns(2)
                with cols[0]:
                    uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration", "csv",
                                                        key="CT_SS_PFCE_Xcal_1"+str(st.session_state.ct))
                    if uploaded_file_Xcal is not None:
                        Xcal = pd.read_csv(uploaded_file_Xcal, index_col=0)
                        wv = np.array(Xcal.columns).astype("float")
                        plotSPC(Xcal, title="NIR spectra - Calibration set - Second instrument")

                with cols[1]:
                    uploaded_file_ycal = st.file_uploader("Upload reference value for calibration", "csv",
                                                        key="CT_SS_PFCE_ycal"+str(st.session_state.ct))
                    if uploaded_file_ycal is not None:
                        ycal = pd.read_csv(uploaded_file_ycal, index_col=0)
                        plotRef_reg(
                            ycal, title="Reference value - Calibration set")

                if "Xcal" in locals() and "ycal" in locals():
                    n_components = st.slider("Number of PLS components", 
                                            1, min([20, int(np.linalg.matrix_rank(Xcal))]), 1)
                    plsModel = pls(n_components=n_components).fit(Xcal, ycal)
                    model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1),
                                          columns=[-1] + list(wv),  index=["Primary model"])

            elif modelSource == "Upload an estabilished model":
                uploaded_file_model = st.file_uploader("Upload your model coefficients", "csv",
                                                    key="CT_SS_PFCE_primary_model_upload"+str(st.session_state.ct))
                st.info("""
                        The uploaded model coefficient file needs to be in csv
                        format. You can download a model file from the
                        Regression page in this site to know format requirement.
                        """)
                if uploaded_file_model is not None:
                    model = pd.read_csv(uploaded_file_model, index_col=0)

        if "model" in locals():
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                plotRegressionCoefficients(model, title="Model coefficients")
    
    # Calibration enhancement with SS-PFCE
    if "X2" in locals() and "y" in locals() and "model" in locals():
        st.header("Calibration enhancement with SS-PFCE")
        st.info(
            """
            The enhanced model can be applied to predict the spectra with  the
            widget of **Predict with regression model** in the **Utils** page of
            this site.
            """
        )

        with st.spinner(
        """
        Calibration enhancement in progress...This will take from a few minutes
        to several minutes depending on the number of variables in NIR spectra.
        """):
            SS_PFCE_model = SS_PFCE(thres=threshould, constrType=constType).fit(
                X2.to_numpy(), y.to_numpy(), model.to_numpy().flatten())
            
        _, col1, _ = st.columns([1, 2, 1])
        with col1:
            slaveModel = pd.DataFrame(data=np.reshape(SS_PFCE_model.b2.x, (1, -1)), 
                                        columns=[-1] + list(wv), index=["Slave model"])
            plotRegressionCoefficients(slaveModel, title="Slave model enhanced by NS-PFCE")


def FS_PFCE_fun(constType="Corr", threshould=0.98):
    # Obtain the data required by FS-PFCE
    st.header("Data required by FS-PFCE")
    st.info(
        """
        The FS-PFCE method requires spectra measured on both the primary/master
        and second/slave instruments (Xm and Xs), as well as the corresponding
        reference values (y) as inputs. It is important to note that the format
        of Xm, Xs and y should comply with the format requirements of this
        platform. Specifically, the samples should be arranged in rows in both
        the spectral and reference value files, with the first column and row
        representing the sample name and wavelength/characteristics of
        "Reference value", respectively.
        """
    )
    dataSource = st.radio("Upload data required by SS-PFCE or use our example data.",
                        ["Tablet", "Upload data manually"], horizontal=True,
                        on_change=changeCT_state,  key="calibration_transfer_FS_PFCE_data_selection")
    
    if dataSource == "Tablet":
        st.info(
            """
            The example dataset provided for calibration transfer consists of
            NIR spectra of 655 pharmaceutical tablets measured on two NIR
            instruments in the wavelength range of 600-1898 nm with a digital
            interval of 2 nm. The noise region of 1794-1898 nm and 13 outlier
            samples were removed as suggested in the [PFCE
            article](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637?via%3Dihub).

            For each template sample, the active pharmaceutical ingredient (API)
            has been measured as the target of calibration. The remaining 642
            samples were divided into a calibration set of 400 samples, a
            standard set of 30 samples, and a prediction set of 212 samples.

            
            To download the example dataset, please use the "example dataset"
            function on the utils page of this site. The dataset has already
            been preprocessed and divided into the calibration, standard, and
            prediction sets as described above. We hope that this dataset will
            be useful to you in your research, and we encourage you to cite the
            relevant literature if you use it in your work. 
            """)
        data = get_Tablet()
        X1 = data["Trans"]["X"][0]
        X2 = data["Trans"]["X"][1]
        y = data["Trans"]["y"]
        wv = data["wv"].flatten()
        X1 = pd.DataFrame(X1, columns=wv)
        X2 = pd.DataFrame(X2, columns=wv)
        y = pd.DataFrame(y, columns=["Reference"])

    elif dataSource == "Upload data manually":
        cols = st.columns(3)
        with cols[0]:
            uploaded_file0 = st.file_uploader(
                "Upload the standard spectra from the first/master instrument", "csv",
                key="CT_FS_PFCE_Xstd_1"+str(st.session_state.ct))
            if uploaded_file0 is not None:
                X1 = pd.read_csv(uploaded_file0, index_col=0)
                wv = np.array(X1.columns).astype("float")
        with cols[1]:
            uploaded_file1 = st.file_uploader(
                "Upload the standard spectra from the second/slave instrument", "csv",
                key="CT_FS_PFCE_Xstd_2"+str(st.session_state.ct))
            if uploaded_file1 is not None:
                X2 = pd.read_csv(uploaded_file1, index_col=0)
        with cols[2]:
            uploaded_file2 = st.file_uploader(
                "Upload the reference values for the standard", "csv",
                key="CT_FS_PFCE_ystd"+str(st.session_state.ct))
            if uploaded_file2 is not None:
                y = pd.read_csv(uploaded_file2, index_col=0)

    cols = st.columns(3)
    if "X1" in locals():
        with cols[0]:
            plotSPC(X1, title="NIR spectra of strandard samples from the first/master instrument")
    if "X2" in locals():
        with cols[1]:
            plotSPC(X2, title="NIR spectra of strandard samples from the second/slave instrument")
    if "y" in locals():
        with cols[2]:
            plotRef_reg(y, title="Reference values of standard samples")



    # Obtain the primary/master model
    if "X1" in locals() and "X2" in locals() and "y" in locals():
        st.header("Primary/master model")
        st.info(
            """
            For PFCE, the model of primary instruments are used as initial
            values to estabilshed the model for second instruments. Therefore,
            you need to build a model via PLS or upload an already built model.
            There is an existing primary model for teblet dataset.
            """)
        if dataSource == "Tablet":
                n_components = 3
                plsModel = pls(n_components=n_components).fit(data["Cal"]["X"][0], data["Cal"]["y"])
                model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1), columns=[-1] + list(wv), index=["Primary model"])

        elif dataSource == "Upload data manually":
            modelSource = st.radio("build or use your own primary model",
                                ["built primary model",
                                    "Upload an estabilished model"],
                                horizontal=True, key="CT_FS_PFCE_primary_model_selection")

            if modelSource == "built primary model":
                cols = st.columns(2)
                with cols[0]:
                    uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration", "csv",
                                                        key="CT_FS_PFCE_Xcal_1"+str(st.session_state.ct))
                    if uploaded_file_Xcal is not None:
                        Xcal = pd.read_csv(uploaded_file_Xcal, index_col=0)
                        wv = np.array(Xcal.columns).astype("float")
                        plotSPC(Xcal, title="NIR spectra - Calibration set - Second instrument")

                with cols[1]:
                    uploaded_file_ycal = st.file_uploader("Upload reference value for calibration", "csv",
                                                        key="CT_FS_PFCE_ycal"+str(st.session_state.ct))
                    if uploaded_file_ycal is not None:
                        ycal = pd.read_csv(uploaded_file_ycal, index_col=0)
                        plotRef_reg(
                            ycal, title="Reference value - Calibration set")

                if "Xcal" in locals() and "ycal" in locals():
                    n_components = st.slider("Number of PLS components", 
                                            1, min([20, int(np.linalg.matrix_rank(Xcal))]), 1)
                    plsModel = pls(n_components=n_components).fit(Xcal, ycal)
                    model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1),
                                          columns=[-1] + list(wv),  index=["Primary model"])

            elif modelSource == "Upload an estabilished model":
                uploaded_file_model = st.file_uploader("Upload your model coefficients", "csv",
                                                    key="CT_FS_PFCE_primary_model_upload"+str(st.session_state.ct))
                st.info("""
                        The uploaded model coefficient file needs to be in csv
                        format. You can download a model file from the
                        Regression page in this site to know format requirement.
                        """)
                if uploaded_file_model is not None:
                    model = pd.read_csv(uploaded_file_model, index_col=0)

        if "model" in locals():
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                plotRegressionCoefficients(model, title="Model coefficients")
    
    # Calibration enhancement with FS-PFCE
    if "X1" in locals() and "X2" in locals() and "y" in locals() and "model" in locals():
        st.header("Calibration enhancement with FS-PFCE")
        st.info(
            """
            The enhanced model can be applied to predict the spectra with  the
            widget of **Predict with regression model** in the **Utils** page of
            this site.
            """
        )

        with st.spinner(
        """
        Calibration enhancement in progress...This will take from a few minutes
        to several minutes depending on the number of variables in NIR spectra.
        """):
            FS_PFCE_model = FS_PFCE(thres=threshould, constrType=constType).fit(
                X1.to_numpy(), X2.to_numpy(), y.to_numpy(), model.to_numpy().flatten())
            
        _, col1, _ = st.columns([1, 2, 1])
        with col1:
            slaveModel = pd.DataFrame(data=np.reshape(FS_PFCE_model.b2.x, (1, -1)), 
                                        columns=[-1] + list(wv), index=["Slave model"])
            plotRegressionCoefficients(slaveModel, title="Slave model enhanced by NS-PFCE")



def MT_PFCE_fun(constType="Corr", threshould=0.98, ntask = 2):
    # Obtain the data required by MT-PFCE
    st.header("Data required by MT-PFCE")
    st.info(
        """
        The MT-PFCE method is designed to enhance calibration involving multiple
        instruments/scenarios. Each instrument/scenario has a set of spectra (Xi)
        and reference values (yi) as input. The spectra and reference values can
        come from different samples. The format of Xi and yi should comply with
        the format requirements of this platform. Specifically, the samples should
        be arranged in rows in both the spectral and reference value files, with
        the first column and row representing the sample name and wavelength/
        characteristics of "Reference value", respectively.
        """
    )

    dataSource = st.radio("Upload data required by SS-PFCE or use our example data.",
                        ["Tablet", "Upload data manually"], horizontal=True,
                        on_change=changeCT_state,  key="calibration_transfer_MT_PFCE_data_selection")
    
    if dataSource == "Tablet":
        st.info(
            """
            The example dataset provided for calibration transfer consists of
            NIR spectra of 655 pharmaceutical tablets measured on two NIR
            instruments in the wavelength range of 600-1898 nm with a digital
            interval of 2 nm. The noise region of 1794-1898 nm and 13 outlier
            samples were removed as suggested in the [PFCE
            article](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637?via%3Dihub).

            For each template sample, the active pharmaceutical ingredient (API)
            has been measured as the target of calibration. The remaining 642
            samples were divided into a calibration set of 400 samples, a
            standard set of 30 samples, and a prediction set of 212 samples.

            
            To download the example dataset, please use the "example dataset"
            function on the utils page of this site. The dataset has already
            been preprocessed and divided into the calibration, standard, and
            prediction sets as described above. We hope that this dataset will
            be useful to you in your research, and we encourage you to cite the
            relevant literature if you use it in your work. 
            """)
        data = get_Tablet()
        wv = data["wv"].flatten()
        Xstd = [pd.DataFrame(data["Cal"]["X"][0], columns=wv),
                pd.DataFrame(data["Trans"]["X"][1], columns=wv)]
        ystd = [pd.DataFrame(data["Cal"]["y"], columns=["Reference"]), 
                pd.DataFrame(data["Trans"]["y"], columns=["Reference"])]
        ntask = 2

    elif dataSource == "Upload data manually":
        Xstd = [None] * ntask
        ystd = [None] * ntask
        for i in range(ntask):
            st.subheader(f"Data for {i+1}th task")
            cols = st.columns(2)
            with cols[0]:
                uploaded_file1 = st.file_uploader(f"Upload the standard spectra from {i}th task", "csv",
                    key="CT_MT_PFCE_Xstd"+str(i)+str(st.session_state.ct))
                if uploaded_file1 is not None:
                    Xi = pd.read_csv(uploaded_file1, index_col=0)
                    wv = np.array(Xi.columns).astype("float")
                    Xstd[i] = Xi
                    plotSPC(Xstd[i], title=f"NIR spectra for the {i+1}th task")
            with cols[1]:
                uploaded_file2 = st.file_uploader(f"Upload the reference values for the {i}th task", "csv",
                    key="CT_MT_PFCE_ystd"+str(i)+str(st.session_state.ct))
                if uploaded_file2 is not None:
                    yi = pd.read_csv(uploaded_file2, index_col=0)
                    ystd[i] = yi
                    plotRef_reg(ystd[i], title=f"Reference values for the {i+1}th task")

    # Obtain the primary/master model
    if len(Xstd) == ntask and len(ystd) == ntask and all(x is not None for x in Xstd) and all(y is not None for y in ystd):
        st.header("Primary/master model")
        st.info(
            """
            For PFCE, the model of primary instruments are used as initial
            values to estabilshed the model for second instruments. Therefore,
            you need to build a model via PLS or upload an already built model.
            There is an existing primary model for teblet dataset.
            """)
        if dataSource == "Tablet":
                n_components = 3
                plsModel = pls(n_components=n_components).fit(data["Cal"]["X"][0], data["Cal"]["y"])
                model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1), columns=[-1] + list(wv), index=["Primary model"])

        elif dataSource == "Upload data manually":
            modelSource = st.radio("build or use your own primary model",
                                ["built primary model",
                                    "Upload an estabilished model"],
                                horizontal=True, key="CT_MT_PFCE_primary_model_selection")

            if modelSource == "built primary model":
                cols = st.columns(2)
                with cols[0]:
                    uploaded_file_Xcal = st.file_uploader("Upload spectra for calibration", "csv",
                                                        key="CT_MT_PFCE_Xcal_1"+str(st.session_state.ct))
                    if uploaded_file_Xcal is not None:
                        Xcal = pd.read_csv(uploaded_file_Xcal, index_col=0)
                        wv = np.array(Xcal.columns).astype("float")
                        plotSPC(Xcal, title="NIR spectra - Calibration set - Second instrument")

                with cols[1]:
                    uploaded_file_ycal = st.file_uploader("Upload reference value for calibration", "csv",
                                                        key="CT_MT_PFCE_ycal"+str(st.session_state.ct))
                    if uploaded_file_ycal is not None:
                        ycal = pd.read_csv(uploaded_file_ycal, index_col=0)
                        plotRef_reg(
                            ycal, title="Reference value - Calibration set")

                if "Xcal" in locals() and "ycal" in locals():
                    n_components = st.slider("Number of PLS components", 
                                            1, min([20, int(np.linalg.matrix_rank(Xcal))]), 1)
                    plsModel = pls(n_components=n_components).fit(Xcal, ycal)
                    model = pd.DataFrame(data=plsModel.model['B'][:, -1].reshape(1,-1),
                                          columns=[-1] + list(wv),  index=["Primary model"])

            elif modelSource == "Upload an estabilished model":
                uploaded_file_model = st.file_uploader("Upload your model coefficients", "csv",
                                                    key="CT_MT_PFCE_primary_model_upload"+str(st.session_state.ct))
                st.info("""
                        The uploaded model coefficient file needs to be in csv
                        format. You can download a model file from the
                        Regression page in this site to know format requirement.
                        """)
                if uploaded_file_model is not None:
                    model = pd.read_csv(uploaded_file_model, index_col=0)

        if "model" in locals():
            _, col1, _ = st.columns([1, 2, 1])
            with col1:
                plotRegressionCoefficients(model, title="Model coefficients")
    
    # Calibration enhancement with MT-PFCE
    if len(Xstd) == ntask and len(ystd) == ntask and all(x is not None for x in Xstd) and all(y is not None for y in ystd) and "model" in locals():
        st.header("Calibration enhancement with MT-PFCE")
        st.info(
            """
            The enhanced model can be applied to predict the spectra with  the
            widget of **Predict with regression model** in the **Utils** page of
            this site.
            """
        )

        with st.spinner(
        """
        Calibration enhancement in progress...This will take from a few minutes
        to several minutes depending on the number of variables in NIR spectra.
        """):
            MT_PFCE_model = MT_PFCE(thres=threshould, constrType=constType).fit(
                [Xi.to_numpy() for Xi in Xstd],
                [yi.to_numpy() for yi in ystd],
                model.to_numpy().flatten())
            
        cols = st.columns(ntask)
        for i in range(ntask):
            with cols[i]:
                slaveModel = pd.DataFrame(data=np.reshape(((MT_PFCE_model.B.x).reshape(ntask,-1).transpose()[:,i]), (1,-1)), 
                                            columns=[-1] + list(wv), index=[f"{i+1}th model"])
                plotRegressionCoefficients(slaveModel, title=f"{i+1}th model enhanced by MT-PFCE")



# Page content
st.set_page_config(page_title="NIR Online-Calibration Enhancement",
                   page_icon=":rocket:", layout="centered")

if 'ct' not in st.session_state:
    st.session_state.ct = 0

st.markdown("# Calibration Transfer/Enhancement")
st.info(
    """
    Parameter free calibration enhancement (PFCE)  is a formal unified NIR
    spectral model enhancement framework proposed by our team that can cope with
    many different known conditions without complex hyperparameter optimization.
    The framework includes four main algorithms, nonsupervised(NS-),
    semisupervised(SS-) , fullsupervised(FS-) and multitask(MT-) PFCE. For more
    information, please refer to this recent published [Article
    1](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637)
    and [Article
    2](https://www.sciencedirect.com/science/article/abs/pii/S0003267020311107).
    """
)

method = st.radio("Select a method",
                  ["NS-PFCE", "SS-PFCE", "FS-PFCE", "MT-PFCE"],
                  horizontal=True)
constType = st.radio("Constraint Type", ["Corr", "L2", "L1"], horizontal=True)

threshould = st.slider("Constraint threshould", 0.00, 1.00, 0.98)

ConstMap = {"Corr": 1, "L2": 2, "L1": 3}
if method == "NS-PFCE":
    NS_PFCE_fun(constType=ConstMap[constType], threshould=threshould)

elif method == "SS-PFCE":
    SS_PFCE_fun(constType=ConstMap[constType], threshould=threshould)

elif method == "FS-PFCE":
    FS_PFCE_fun(constType=ConstMap[constType], threshould=threshould)

elif method == "MT-PFCE":
    ntask = st.slider("Number of tasks", 2, 4, 2)
    MT_PFCE_fun(constType=ConstMap[constType], threshould=threshould, ntask = ntask)
