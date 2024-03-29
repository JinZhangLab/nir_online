import streamlit as st
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from pynir.Calibration import regressionReport
from collections import Counter
from sklearn.metrics import r2_score, mean_squared_error

from .dataManipulation import download_img, download_csv

plt.rcParams['pdf.fonttype'] = 42

# Common plots


def plotSPC(X, title="NIR spectra"):
    """
    Plots the NIR spectra for a given set of samples.

    Args:
    X (pandas.DataFrame): A DataFrame containing the NIR spectra for each sample. The index should contain the sample names and the columns should contain the wavelength values.
    title (str): The title of the plot. Default is "NIR spectra".

    Returns:
    None
    """
    sampleName = X.index.to_numpy(dtype=str)
    wv = X.columns.to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.plot(wv, np.transpose(X))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (A.U.)')
    ax.set_title(title, loc='center')
    fig.tight_layout()

    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="NIRSpectra", label="Download image")
        download_csv(X, index = True, columns=True,  index_label="Sample name\\Wavelength (nm)",
                     fileName="Spectra", label="Download spectral file")

def pltPCAscores_2d(scores, Vars = None, title="PCA scores"):
    """
    Plots the PCA scores for a given set of samples in 2D.

    Args:
    scores (pandas.DataFrame): A DataFrame containing the PCA scores for each sample. The index should contain the sample names and the columns should contain the PCA scores.
    Vars (list): A list containing the explained variance ratio for the first two principal components. Default is None.
    title (str): The title of the plot. Default is "PCA scores".

    Returns:
    None
    """

    fig, ax = plt.subplots()
    ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1])

    # plot 95 % confidence eclipse
    xmean = np.mean(scores.iloc[:, 0])
    ymean = np.mean(scores.iloc[:, 1])
    xstd = np.std(scores.iloc[:, 0])
    ystd = np.std(scores.iloc[:, 1])
    cov = np.cov(scores.iloc[:, 0], scores.iloc[:, 1])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = mpl.patches.Ellipse(xy=(xmean, ymean),
                                width=lambda_[0] * 2 * 1.96, height=lambda_[1] * 2 * 1.96,
                                angle=np.rad2deg(np.arccos(v[0, 0])), color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)

    ax.set_xlabel(f"PC1 ({round(Vars[0]*100,2)}%)" if Vars is not None else "PC1")
    ax.set_ylabel(f"PC2 ({round(Vars[1]*100,2)}%)" if Vars is not None else "PC2")

    ax.set_title(title, loc='center')
    fig.tight_layout()

    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="PCAscores", label="Download image")
        download_csv(scores, index = True, columns=True, index_label="Sample name",
                     fileName="PCAscores", label="Download PCA scores file")

def plotRef_reg(y, title = "Reference values"):
    """
    Plots a histogram of the reference values in a pandas DataFrame for regression.

    Args:
        y (pandas.DataFrame): A DataFrame containing the reference values. Each column represents a sample and each row represents a variable.

    Returns:
        None.

    Displays a histogram of the reference values in a Streamlit tab along with buttons to download the plot as an image and the reference values as a CSV file.
    """
    fig, ax = plt.subplots()
    ax.hist(y, rwidth=0.9)
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Count')
    ax.set_title(title, loc='center')

    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="ReferenceValues", label="Download image")
        download_csv(y,  index = True, columns=True, index_label="Sample name", 
                     fileName="Reference", label="Download reference value file")


def plotRef_clf(y):
    """
    Plots a bar chart of the reference values in a pandas DataFrame for classification.

    Args:
        y (pandas.DataFrame): A DataFrame containing the reference values. The first column represents the class labels and each row represents a sample.

    Returns:
        None.

    Displays a bar chart of the reference values in a Streamlit tab along with buttons to download the plot as an image and the reference values as a CSV file.
    """

    mycount = Counter(y.iloc[:, 0])
    fig, ax = plt.subplots()
    ax.bar(list(mycount.keys()), list(mycount.values()))
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title('Reference values', loc='center')

    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="ReferenceValues", label="Download image")
        download_csv(y,  index = True, columns=True, index_label="Sample name", 
                     fileName="Reference", label="Download reference value file")



def plotFOMvsHP(FOM, xlabel="Hypers", ylabel="FOM", title="NIR online", cmap=None, markers=None):
    """ Figure of merit vs latent variable 
    Input:
    FOM, DataFrame, each row contains a set of figure of merits at different hyperparameters.
    for example:
    FOM = pd.DataFrame(
        data=[[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.3, 0.4]],
        index = ["RMSEC", "RMSECV"], columns = [1, 2, 3, 4])
    """
    fig, ax = plt.subplots()
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    if markers is None:
        markers = ["o", "v", "s", "p", "P", "*", "X", "D", "d", "h"]
    for i, row in enumerate(FOM.index):
        ax.plot(FOM.columns, FOM.loc[row, :], color=cmap(i),
                marker=markers[i], label=row)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='center')
    ax.legend()

    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="FOM", label="Download image")
        download_csv(FOM,  index = True, columns=True, index_label="FOM\\Hypers", 
                     fileName="FOM", label="Download file")

def plotPrediction_reg(y, xlabel="Reference", ylabel="Prediction", title="NIR online", cmap=None, markers=None):
    """ Prediction vs reference plot
    Input:
    y, DataFrame, the first row contains reference values, each row from the second to the last are predictions by each method
    for example:
    y = pd.DataFrame(
        data=[[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.3, 0.4], [0.2, 0.3, 0.2, 0.4]],
        columns = ["Reference", "method1", "methods2"], index = ["Sample1","Sample2","Sample3"])
    """
    fig, ax = plt.subplots()
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    if markers is None:
        markers = ["o", "v", "s", "p", "P", "*", "X", "D", "d", "h"]
    RMSE = []
    R2 = []
    for i, col in enumerate(y.columns):
        if i==0:
            ax.plot([np.min(y.iloc[:, 0])*0.95, np.max(y.iloc[:, 0])*1.05], [np.min(y.iloc[:, 0])*0.95, np.max(y.iloc[:, 0])*1.05], color='black', label="y=x")
        else:
            ax.scatter(y.iloc[:, 0], y.iloc[:, i], color=cmap(i), marker=markers[i], label=col)
            RMSE.append(np.sqrt(mean_squared_error(y.iloc[:, 0], y.iloc[:, i])))
            R2.append(r2_score(y.iloc[:, 0], y.iloc[:, i])) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='center')
    # RMSE and R2 for all methods, not only the first one, or the average of all methods
    textstr = "\n".join([f"{col}: RMSE={rmse:.2f}, R2={r2:.2f}" for col, rmse, r2 in zip(y.columns[1:], RMSE, R2)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.4, 0.2, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    ax.legend()

    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="Prediction", label="Download image")
        download_csv(y,  index = True, columns=True, index_label="Sample name", 
                     fileName="Prediction", label="Download file")

def plotRegressionCoefficients(b, title = "Regression coefficients", index_label="Wavelength (nm)"):
    wv = np.array(b.columns, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(wv[1:], b.to_numpy().ravel()[1:])
    ax.set_xlabel("Wavenumber (nm)")
    ax.set_ylabel("Coefficients")
    ax.set_title(title)
    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="Regression Coefficients", label="Download image")
        download_csv(b,  index = True, columns=True, index_label=index_label,
                     fileName="Regression Coefficients", label="Download model")

# plot for classfication by plsda
# plot rmsecv variation against lv
def plotAccuracyCV(accuracy_cv, labels='accuracy'):

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(accuracy_cv))+1, accuracy_cv, color='tab:orange',
            marker='.', label=labels)
    # ax.plot(optLV, r2cv[optLV-1],marker='o',markersize=10,label='optimal LV')
    ax.set_xlabel("Number of pls components")
    ax.set_ylabel(labels)
    ax.set_title("Variation of "+labels +
                 " with nLV in calibration and cross validation")
    ax.legend()
    st.pyplot(fig)
    download_img(fig, fileName="AccuracyCV")


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues


    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    fig, ax = plt.subplots()

    accuracy = np.trace(cm.to_numpy()) / np.sum(cm.to_numpy()).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm.to_numpy(), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.to_numpy().max() / 2
    for i in range(cm.to_numpy().shape[0]):
        for j in range(cm.to_numpy().shape[1]):
            plt.text(j, i, "{:,}".format(cm.to_numpy()[i, j]),
                        horizontalalignment="center",
                        color="white" if cm.to_numpy()[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName=title,label="Download image")
        download_csv(cm, index=True, columns=True, index_label="True label\\Prediction",
                      fileName=title, label="Download csv")


# Feature Selection
def plotVariableImportance(Imp, xlabel="Wavelength (nm)", ylabel = "Intensity", title = "Variable importance"):
    """
    Imp: pandas dataframe with variable importance.
         The row correspond the method, the column correspond the variable importance for each variable.
         The variable importance by each variable are normalized to the range of 0-1 for comparison.
    """
    fig, ax = plt.subplots()
    wv = np.array(Imp.columns, dtype=float)
    for i in range(Imp.shape[0]):
        impi = (Imp.iloc[i, :]-Imp.iloc[i, :].min()) / (Imp.iloc[i, :].max()-Imp.iloc[i, :].min())
        ax.plot(wv, impi, label=Imp.index[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="Variable Importance", label="Download image")
        download_csv(Imp,  index = True, columns=True, index_label="Method\\Wavelength (nm)",
                     fileName="Variable Importance", label="Download file")


def plotVariableSelection(X, FS, xlabel="Wavelength (nm)", ylabel = "Intensity", title = "Variable importance"):
    """
    X: pandas dataframe with spectra.
    FS: pandas dataframe with variable selection result.
         The row correspond the method, the column correspond that the variable is selected or not for each variable.
    """
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, )
    wv = np.array(X.columns, dtype=float)
    ax1.plot(wv, X.to_numpy().T)
    ax1.set_ylabel(ylabel)
    ax1.set_title("Variable selection")

    for i in range(FS.shape[0]):
        fsi = FS.iloc[i, :].to_numpy(dtype=bool)
        ax2.eventplot(wv[fsi], lineoffsets=i+1, linelengths=0.5, linewidths=1.0)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Method")
    ax2.set_yticks(np.arange(FS.shape[0])+1)
    ax2.set_yticklabels(FS.index)

    plt.subplots_adjust(hspace=0)

    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.spines['bottom'].set_linestyle(' ') 
    ax2.spines['top'].set_linestyle(' ')
    ax2.set_ylim(ax2.get_ylim()[::-1])

    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="Variable Selection", label="Download image")
        download_csv(FS,  index = True, columns=True, index_label="Method\\Wavelength (nm)",
                     fileName="Variables Selected", label="Download file")