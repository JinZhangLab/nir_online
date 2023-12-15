# Import modules
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pynir.Calibration import sampleSplit_KS
from pynir.utils import simulateNIR
from tools.display import plotSPC, plotRef_reg, plotRef_clf, pltPCAscores_2d, plotRegressionCoefficients, plotPrediction_reg
from tools.dataManipulation import download_csv, download_csv, get_Tablet, get_PlantLeaf, get_Corn, predict_reg
from sklearn.model_selection import train_test_split

# Simulate NIR Data


def simulate_nir_data():
    """Simulate NIR data with user-defined parameters."""
    st.markdown("### Set simulation parameters")
    n_components = st.slider('Number of components', 2, 20, 10)
    n_samples = st.slider('Number of samples', 10, 2000, 100)
    noise_level = st.slider('Noise level (*10^-5)', 0.00, 10.00, 1.00)
    seeds = st.slider('Random seeds', 0, 10000, 0)
    ref_type = st.slider('Reference value type', 1, min([5, round(n_samples/2)]), 1,
                         help="""
                         The `Reference value type` parameter specifies the type
                         of reference values to use in the model. If it is 1,
                         the reference values are resamplesd from a continuous
                         region and used for regression. If it is an integer
                         greater than 1, the reference values belong to the
                         corresponding number of classes and are used for binary
                         or multiple classification.
                         """)

    X, y, wv = simulateNIR(nSample=n_samples,
                           n_components=n_components,
                           noise=noise_level*(10**-5),
                           refType=ref_type,
                           seeds=seeds)
    sampleName = [f"Sample_{i}" for i in range(n_samples)]
    X = pd.DataFrame(X, index=sampleName, columns=wv)
    y = pd.DataFrame(y, index=sampleName, columns=["Reference value"])
    cols = st.columns(2)
    with cols[0]:
        plotSPC(X)

    with cols[1]:
        if ref_type == 1:
            plotRef_reg(y)
        else:
            plotRef_clf(y)



# prediction with regression model
def predict_with_regression_model():
    """
    Allows the user to upload a regression model and spectra to be predicted,
    and generates predictions using the model.

    The function prompts the user to upload a regression model and spectra to be
    predicted in CSV format. If both files are uploaded successfully, the
    function generates predictions using the regression model and displays the
    predicted reference values in a DataFrame. The user can also upload
    reference values for the spectra to be predicted and compare them to the
    predicted values using a plot.

    Returns:
        None
    """
    st.markdown("## Predict with Regression Model")
    st.markdown("Upload a regression model and spectra to be predicted in CSV format.")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_model = st.file_uploader("Upload regression model", type="csv")
        if uploaded_model is not None:
            model_reg = pd.read_csv(uploaded_model, index_col=0)
            plotRegressionCoefficients(model_reg)
    with col2:
        uploaded_spectra = st.file_uploader("Upload spectra to be predicted", type="csv")
        if uploaded_spectra is not None:
            X = pd.read_csv(uploaded_spectra, index_col=0)
            plotSPC(X, title="Spectra to be predicteds")
            
    if "model_reg" in locals() and "X" in locals():
        y_pred = predict_reg(X, model_reg)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("## Predicted Reference Values")
            st.markdown("The predicted reference values for the uploaded spectra are:")
            st.dataframe(y_pred)
        with col2:
            st.markdown("## Compare Predictions to Reference Values")
            st.markdown("Upload your reference values for the spectra to be predicted for comparison.")
            uploaded_ref = st.file_uploader("Upload reference values", type="csv")
            if uploaded_ref is not None:
                y = pd.read_csv(uploaded_ref, index_col=0)
                y_to_plot = pd.DataFrame(data=np.concatenate((y.to_numpy(), y_pred.to_numpy()), axis=1),
                                         columns=["Reference values", "Predictions"], index=y.index)
                plotPrediction_reg(y_to_plot)

# Split Sample Set
def split_sample_set():
    """Split sample set into calibration and validation set."""
    st.markdown("""
    ## Split sample set into calibration and validation set
    Two methods are supported to split the sample set now: random and Kennard-Stone (KS) algorithm.
    """)
    split_method = st.radio("Split method", ("Random", "KS"),
                            on_change=st.cache_data.clear())

    if split_method == "Random":
        num_samples = st.slider("Number of samples", 10, 2000, 100)
        split_ratio = st.slider("Train ratio", 0.1, 0.9, 0.8)
        seeds = st.slider("Random seeds", 0, 10000, 0)

        sampleIdx = np.arange(num_samples)
        sampleName = np.array([f"Sample_{i}" for i in range(num_samples)])
        trainIdx, testIdx = train_test_split(sampleIdx,
                                             test_size=round(
                                                 num_samples * (1 - split_ratio)),
                                             random_state=seeds,
                                             shuffle=True)
        trainIdx = pd.DataFrame(data=trainIdx,
                                index=sampleName[trainIdx],
                                columns=["Train set index"])
        testIdx = pd.DataFrame(data=testIdx,
                               index=sampleName[testIdx],
                               columns=["Test set index"])
        col1, col2 = st.columns(2)
        with col1:
            st.write(trainIdx)
        with col2:
            st.write(testIdx)
    elif split_method == "KS":
        st.markdown("""
        The KS algorithms aim to select a subset of samples whose spectra are as
        different from each other as possible. Therefore, your spectra should be
        uploaded as a CSV file with samples in rows and wavenumber in columns.
        """)
        uploaded_file = st.file_uploader("Upload your spectra here", "csv")
        if uploaded_file is not None:
            X = pd.read_csv(uploaded_file, index_col=0)
            sampleName = X.index.to_numpy(dtype=str)
            wv = X.columns.to_numpy(dtype=float)

            scores = PCA(n_components=2).fit_transform(X)
            scores = pd.DataFrame(data=scores,
                                  index=sampleName,
                                  columns=["PC1", "PC2"])
            pltPCAscores_2d(scores=scores,
                            title="PCA scores of samples")

            split_ratio = st.slider("Train ratio", 0.1, 0.9, 0.8)
            trainIdx, testIdx = sampleSplit_KS(X.to_numpy(),
                                               test_size=1 - split_ratio)
            trainIdx = pd.DataFrame(data=trainIdx,
                                    index=sampleName[trainIdx],
                                    columns=["Train set index"])
            testIdx = pd.DataFrame(data=testIdx,
                                   index=sampleName[testIdx],
                                   columns=["Test set index"])
            col1, col2 = st.columns(2)
            with col1:
                st.write(trainIdx)
            with col2:
                st.write(testIdx)

    if "trainIdx" in locals() and "testIdx" in locals():
        st.markdown("## Apply the split to your spectral or reference value data: ")
        uploaded_file = st.file_uploader("Upload file to be split here", "csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, index_col=0)
            col1, col2 = st.columns(2)
            with col1:
                download_csv(data.iloc[trainIdx.iloc[:, 0], :],
                             index=True,
                             columns=True,
                             fileName="trainSet",
                             label="Download train set")
            with col2:
                download_csv(data.iloc[testIdx.iloc[:, 0], :],
                             index=True,
                             columns=True,
                             fileName="testSet",
                             label="Download test set")

# Convert WaveNumber (cm-1) to Wavelength (nm)


def convert_wn_to_wl():
    """Convert the wavenumber in a spectral file with WaveNumber (cm-1) unit to
    Wavelength (nm)."""
    st.markdown("""
    Convert the wavenumber in a spectral file with WaveNumber (cm-1) unit to
    Wavelength (nm). Before conversion, be sure that the first row of your
    spectral file is the wavenumber with the unit of cm^-^1.
    """)
    uploaded_file = st.file_uploader("Upload your spectra here", "csv")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file, index_col=0)
        sampleName = X.index.to_numpy(dtype=str)
        wv = X.columns.to_numpy(dtype=float)
        wv = 1e7 / wv
        X = pd.DataFrame(data=X.to_numpy(), index=sampleName, columns=wv)
        download_csv(X,
                     index=True,
                     columns=True,
                     index_label="Sample Name\\Wavelength (nm)",
                     fileName="Spectra_converted",
                     label="Download converted spectra")

# Transpose data in CSV file


def transpose_data():
    """Transpose the row and column in a CSV file."""
    st.markdown("""
    You can use this function to transpose the row and column in a CSV file.
    """)
    uploaded_file = st.file_uploader("Upload your file here", "csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0)
        data = data.transpose()
        download_csv(data,
                     index=True,
                     columns=True,
                     fileName="Data_transposed",
                     label="Download transposed data")

# Read Spectra from files


def read_spectra_from_files():
    """Read spectral data from raw files obtained from a spectrometer."""
    st.markdown("""
    This function allows you to read spectral data from raw files obtained from
    a spectrometer. Please note that we currently only support spectral files
    from DMD spectrometers manufactured by companies such as
    http://www.ias-nir.com/.

    To use this function, simply upload your spectral files in CSV format using
    the file uploader. The function will then read the spectral data from the
    files and display it in a pandas DataFrame. You can choose to average the
    spectra of replicates by selecting the "Average the spectra of replicates"
    checkbox.

    Once the spectral data is displayed, you can download it as a CSV file using
    the "Download spectra" button.
    """)

    # averge the spectra of replicates with file name ended with "_1.csv", "_2.csv", "_3.csv", etc.

    mean_replicates = st.checkbox(
        "Average the spectra of replicates",
        help="""
            Average the spectra of replicates with file name ended with
            '_1.csv', '_2.csv', '_3.csv', etc.
        """)

    with st.expander("Extra settings for spectral reader", expanded=False):
        num_skip_rows = st.number_input("Number of rows to skip", value=18)
        num_index_col = st.number_input("Which column in csv file to use as index", value=0)
        num_spec_col = st.number_input("Which column in dataframe to use as spectra", value=0)

        delimiter_dict = {"Comma (,)": ",",
                          "Semicolon (;)": ";",
                          "Tab": "\t",
                          "Space": " ",
                          "Colon (:)": ":",
                          "Vertical bar (|)": "|",
                          "None": None}
        delimiter = st.selectbox("Delimiter", delimiter_dict.keys(), index=0)
        delimiter = delimiter_dict[delimiter]

    uploaded_files = st.file_uploader(
        "Upload your spectra here", "csv", accept_multiple_files=True)
    if uploaded_files:
        data = pd.DataFrame()
        for file in uploaded_files:
            spci = pd.read_csv(file, index_col=num_index_col, skiprows=num_skip_rows, delimiter=delimiter)
            spci.dropna(axis=1, how="all", inplace=True)
            spci.dropna(axis=0, how="all", inplace=True)
            spci = spci.iloc[:, num_spec_col]
            data = pd.concat([data, pd.DataFrame(
                data=spci.to_numpy(), index=spci.index, columns=[file.name]).transpose()])

        if mean_replicates:
            fileName_mean = []
            for x in data.index.to_numpy():
                if "_" in x:
                    fileName_mean.append("_".join(x.split("_")[:-1]))
                else:
                    fileName_mean.append(x)
            data = data.groupby(fileName_mean).mean()

        st.dataframe(data)
        download_csv(data, index=True, columns=True,
                     fileName="Spectra_converted", label="Download spectra")

# Example dataset


def example_dataset():
    """Download example datasets for NIR calibration."""
    st.markdown("""
    This function enables you to download example datasets for NIR calibration,
    including the Corn dataset, Plant Leaf dataset, and Tablet dataset. These
    datasets are described in detail in the
    [paper](https://www.sciencedirect.com/science/article/abs/pii/S1386142523006637?via%3Dihub).

    Please note that these datasets are divided into different sets based on the
    experimental description in our papers, and are designed to enable you to
    reproduce our results. 
    """)
    dataset_sel = st.radio("Select the dataset you want to download",
                           ("Corn dataset", "Plant Leaf dataset", "Tablet dataset"))
    if dataset_sel == "Corn dataset":
        data = get_Corn()

        wv = data["wv"].flatten()

        X1_cal = pd.DataFrame(data["Cal"]["X"][0], columns=wv)
        X2_cal = pd.DataFrame(data["Cal"]["X"][1], columns=wv)
        X3_cal = pd.DataFrame(data["Cal"]["X"][2], columns=wv)
        ycal = pd.DataFrame(data["Cal"]["y"], columns=["Reference Value (%)"])

        X1_std = pd.DataFrame(data["Trans"]["X"][0], columns=wv)
        X2_std = pd.DataFrame(data["Trans"]["X"][1], columns=wv)
        X3_std = pd.DataFrame(data["Trans"]["X"][2], columns=wv)
        y_std = pd.DataFrame(data["Trans"]["y"], columns=[
                             "Reference Value (%)"])

        X1_test = pd.DataFrame(data["Test"]["X"][0], columns=wv)
        X2_test = pd.DataFrame(data["Test"]["X"][1], columns=wv)
        X3_test = pd.DataFrame(data["Test"]["X"][2], columns=wv)
        y_test = pd.DataFrame(data["Test"]["y"], columns=[
                              "Reference Value (%)"])

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            download_csv(X1_cal,  index=True,  columns=True,  index_label="Sample Name\\Wavelength (nm)",
                            fileName="X1_m5_cal",  label="X-calibration-master")
            download_csv(X1_std,  index=True, columns=True,  index_label="Sample Name\\Wavelength (nm)",
                            fileName="X1_m5_std", label="X-standard-master")
            download_csv(X1_test,  index=True,  columns=True,  index_label="Sample Name\\Wavelength (nm)",
                            fileName="X1_m5_test",  label="X-test-master")
        with col2:
            download_csv(X2_cal,  index=True, columns=True, index_label="Sample Name\\Wavelength (nm)",
                            fileName="X2_mp5_cal", label="X-calibration-slave1")
            download_csv(X2_std,  index=True,  columns=True,  index_label="Sample Name\\Wavelength (nm)",
                            fileName="X2_mp5_std",  label="X-standard-slave1")
            download_csv(X2_test, index=True, columns=True, index_label="Sample Name\\Wavelength (nm)",
                            fileName="X2_mp5_test",  label="X-test-slave1")
        with col3:
            download_csv(X3_cal, index=True,  columns=True, index_label="Sample Name\\Wavelength (nm)",
                            fileName="X3_mp6_cal",  label="X-calibration-slave2")
            download_csv(X3_std, index=True, columns=True, index_label="Sample Name\\Wavelength (nm)",
                            fileName="X3_mp6_std", label="X-standard-slave2")
            download_csv(X3_test, index=True,  columns=True, index_label="Sample Name\\Wavelength (nm)",
                            fileName="X3_mp6_test",  label="X-test-slave2")
        with col4:
            download_csv(ycal, index=True, columns=True, index_label="Sample Name",
                            fileName="ycal", label="y-calibration")
            download_csv(y_std,  index=True, scolumns=True, index_label="Sample Name",
                            fileName="y_std", label="y-standard")
            download_csv(y_test, index=True, columns=True, index_label="Sample Name",
                            fileName="y_test",  label="y-test")
        st.markdown(
            " The master, slave1 and slave 2 refer to the three spectrometers of m5, mp5 and mp6, respectively.")

    elif dataset_sel == "Plant Leaf dataset":
        data = get_PlantLeaf()
        wv = data["wv"].flatten()
        X1_cal = pd.DataFrame(data["Cal"]["X"][0], columns=wv)
        X2_cal = pd.DataFrame(data["Cal"]["X"][1], columns=wv)
        ycal = pd.DataFrame(data["Cal"]["y"], columns=["Reference Value (%)"])

        X1_std = pd.DataFrame(data["Trans"]["X"][0], columns=wv)
        X2_std = pd.DataFrame(data["Trans"]["X"][1], columns=wv)
        y_std = pd.DataFrame(data["Trans"]["y"], columns=[
                             "Reference Value (%)"])

        X1_test = pd.DataFrame(data["Test"]["X"][0], columns=wv)
        X2_test = pd.DataFrame(data["Test"]["X"][1], columns=wv)
        y_test = pd.DataFrame(data["Test"]["y"], columns=[
                              "Reference Value (%)"])

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            download_csv(X1_cal, index=True, columns=True,  index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X1_master_cal", label="X-calibration-master")
            download_csv(X1_std,   index=True,  columns=True,  index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X1_master_std",  label="X-standard-master")
            download_csv(X1_test,  index=True, columns=True,  index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X1_master_test", label="X-test-master")
        with col2:
            download_csv(X2_cal,   index=True,  columns=True, index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X2_slave_cal", label="X-calibration-slave")
            download_csv(X2_std,  index=True, columns=True, index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X2_slave_std", label="X-standard-slave")
            download_csv(X2_test, index=True,  columns=True, index_label="Sample Name\\WaveNumber (cm-1)",
                            fileName="X2_slave_test", label="X-test-slave")
        with col3:
            download_csv(ycal, index=True, columns=True, index_label="Sample Name",
                            fileName="ycal", label="y-calibration")
            download_csv(y_std, index=True, columns=True, index_label="Sample Name",
                            fileName="y_std", label="y-standard")
            download_csv(y_test, index=True,  columns=True, index_label="Sample Name",
                            fileName="y_test", label="y-test")
        st.markdown(
            " The master and slave refer to data measured on samples ground to 40 and 60 mesh, respectively.")

    elif dataset_sel == "Tablet dataset":
        data = get_Tablet()
        wv = data["wv"].flatten()
        X1_cal = pd.DataFrame(data["Cal"]["X"][0], columns=wv)
        X2_cal = pd.DataFrame(data["Cal"]["X"][1], columns=wv)
        ycal = pd.DataFrame(data["Cal"]["y"], columns=["Reference Value (%)"])

        X1_std = pd.DataFrame(data["Trans"]["X"][0], columns=wv)
        X2_std = pd.DataFrame(data["Trans"]["X"][1], columns=wv)
        y_std = pd.DataFrame(data["Trans"]["y"], columns=[
                             "Reference Value (%)"])

        X1_test = pd.DataFrame(data["Test"]["X"][0], columns=wv)
        X2_test = pd.DataFrame(data["Test"]["X"][1], columns=wv)
        y_test = pd.DataFrame(data["Test"]["y"], columns=[
                              "Reference Value (%)"])

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            download_csv(X1_cal, index=True, columns=True, index_label="Sample Name\\WaveLength (nm)",
                            fileName="X1_master_cal", label="X-calibration-master")
            download_csv(X1_std, index=True, columns=True, index_label="Sample Name\\WaveLength (nm)",
                            fileName="X1_master_std", label="X-standard-master")
            download_csv(X1_test, index=True, columns=True,  index_label="Sample Name\\WaveLength (nm)",
                            fileName="X1_master_test",  label="X-test-master")
        with col2:
            download_csv(X2_cal, index=True, columns=True, index_label="Sample Name\\WaveLength (nm)",
                            fileName="X2_slave_cal", label="X-calibration-slave")
            download_csv(X2_std, index=True,  columns=True, index_label="Sample Name\\WaveLength (nm)",
                            fileName="X2_slave_std", label="X-standard-slave")
            download_csv(X2_test, index=True, columns=True, index_label="Sample Name\\WaveLength (nm)",
                            fileName="X2_slave_test", label="X-test-slave")
        with col3:
            download_csv(ycal, index=True, columns=True, index_label="Sample Name",
                            fileName="ycal", label="y-calibration")
            download_csv(y_std, index=True, columns=True, index_label="Sample Name",
                            fileName="y_std", label="y-standard")
            download_csv(y_test, index=True, columns=True, index_label="Sample Name",
                            fileName="y_test", label="y-test")

        st.markdown(
            " The master and slave refer to data measured on the first and second instruments, respectively.")


# Page content
st.set_page_config(page_title="NIR Online-Utils",
                   page_icon=":rocket:", layout="centered")
st.markdown("# Tools useful for NIR calibration")

function_sel = st.radio(
    "Tool collection that may be helpful for NIR calibration.",
    ("Simulate NIR data",
     "Predict with regression model",
     "Split Sample Set",
     "Convert WaveNumber (cm-1) to Wavelength (nm)",
     "Transpose data in CSV file",
     "Read Spectra from files",
     "Example dataset",
     "Others"),
    horizontal=True,
    key="function_selector"
)

st.markdown("---")

if function_sel == "Simulate NIR data":
    simulate_nir_data()

if function_sel == "Predict with regression model":
    predict_with_regression_model()

if function_sel == "Split Sample Set":
    split_sample_set()

if function_sel == "Convert WaveNumber (cm-1) to Wavelength (nm)":
    convert_wn_to_wl()

if function_sel == "Transpose data in CSV file":
    transpose_data()

if function_sel == "Read Spectra from files":
    read_spectra_from_files()

if function_sel == "Example dataset":
    example_dataset()

if function_sel == "Others":
    st.write("""Additional functions will be added to this site in the near
             future. If you have any specific requirements or suggestions,
             please submit an issue on the [GitHub repository](https://github.com/JinZhangLab/nir_online/issues). 
             We appreciate your feedback and will do our best to address your needs.""")
