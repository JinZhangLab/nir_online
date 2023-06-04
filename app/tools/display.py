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

# display reference values for regression
def plotRef_reg(y):
    fig, ax = plt.subplots()
    ax.hist(y, rwidth=0.9)
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Count')
    ax.set_title('Reference values', loc='center')

    tab1, tab2 = st.tabs(["Figure", "Download"])
    with tab1:
        st.pyplot(fig)
    with tab2:
        download_img(fig, fileName="ReferenceValues", label="Download image")
        download_csv(y,  index = True, columns=True, index_label="Sample name", 
                     fileName="Reference", label="Download reference value file")


# display reference values for classification
def plotRef_clf(y):
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
# plot for regression by pls
# plot rmsecv variation against lv


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
        index = ["Reference", "method1", "methods2"], columns = ["Sample1","Sample2","Sample3","Sample4"])
    """
    fig, ax = plt.subplots()
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    if markers is None:
        markers = ["o", "v", "s", "p", "P", "*", "X", "D", "d", "h"]
    RMSE = []
    R2 = []
    for i, row in enumerate(y.index):
        if i==0:
            ax.plot([np.min(y.iloc[0, :])*0.95, np.max(y.iloc[0, :])*1.05], [np.min(y.iloc[0, :])*0.95, np.max(y.iloc[0, :])*1.05], color='black', label="y=x")
        else:
            ax.scatter(y.iloc[0, :], y.iloc[i, :], color=cmap(i), marker=markers[i], label=row)
            RMSE.append(np.sqrt(mean_squared_error(y.iloc[0, :], y.iloc[i, :])))
            R2.append(r2_score(y.iloc[0, :], y.iloc[i, :])) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='center')
    # RMSE and R2 for all methods, not only the first one, or the average of all methods
    textstr = "\n".join([f"{row}: RMSE={rmse:.2f}, R2={r2:.2f}" for row, rmse, r2 in zip(y.index[1:], RMSE, R2)])
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

def plotRegressionCoefficients(b, title = "Regression coefficients"):
    wv = np.array(b.index, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(wv[1:], b.to_numpy()[1:])
    ax.set_xlabel("Wavenumber (nm)")
    ax.set_ylabel("Coefficients")
    ax.set_title(title)
    col1, col2 = st.tabs(["Figure", "Download"])
    with col1:
        st.pyplot(fig)
    with col2:
        download_img(fig, fileName="RegressionCoefficients", label="Download image")
        download_csv(b,  index = True, columns=True, index_label="Wavenumber",
                     fileName="RegressionCoefficients", label="Download file")

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
