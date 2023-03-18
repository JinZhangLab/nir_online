import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pynir.Calibration import regresssionReport
from collections import Counter

from .dataManipulation import download_img

plt.rcParams['pdf.fonttype'] = 42

# Common plots
def plotSPC(X,wv=None, title = "NIR spectra"):
    if wv is None:
        wv = np.arange(X.shape[1])
    wv = np.squeeze(wv)
    fig, ax = plt.subplots()
    if wv.shape[0] == X.shape[0]:
        ax.plot(wv, np.array(X))
    elif wv.shape[0] == X.shape[1]:
        ax.plot(wv, np.transpose(X))
    else:
        raise("The data dimension does not match!")
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (A.U.)')
    ax.set_title(title, loc='center')
    st.pyplot(fig)
    fig.tight_layout()

    download_img(fig, fileName = "NIRSpectra")


    
# display reference values for regression
def plotRef(y):
    fig, ax = plt.subplots()
    ax.hist(y,rwidth=0.9)
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Count')
    ax.set_title('Reference values', loc='center')
    st.pyplot(fig)
    download_img(fig, fileName = "ReferenceValues")
    
# display reference values for classification 
def plotRef_clf(y):
    mycount = Counter(y)
    fig, ax = plt.subplots()
    ax.bar(mycount.keys(),mycount.values())
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Count')
    ax.set_title('Reference values', loc='center')
    st.pyplot(fig)
    download_img(fig, fileName = "ReferenceValues")

## plot for regression by pls
# plot rmsecv variation against lv
def plotRMSECV(rmsec, rmsecv):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rmsecv))+1,rmsec,color = 'tab:blue', 
            marker='*', label ='RMSEC')
    ax.plot(np.arange(len(rmsecv))+1,rmsecv,color = 'tab:orange', 
            marker='.', label ='RMSECV')
    #ax.plot(optLV, rmsecv[optLV-1],marker='o',markersize=10,label='optimal LV')
    ax.set_xlabel("Number of pls components")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.set_title("Variation of RMSE with nLV in calibration and cross validation")
    st.pyplot(fig)
    download_img(fig, fileName = "RMSECV")

# plot rmsecv variation against lv
def plotR2CV(r2, r2cv):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(r2))+1,r2,color = 'tab:blue', 
            marker='*', label ='R$^2$')
    ax.plot(np.arange(len(r2cv))+1,r2cv,color = 'tab:orange', 
            marker='.', label ='R$^2$$_C$$_V$')
    #ax.plot(optLV, r2cv[optLV-1],marker='o',markersize=10,label='optimal LV')
    ax.set_xlabel("Number of pls components")
    ax.set_ylabel("R$^2$")
    ax.set_title("Variation of R$^2$ with nLV in calibration and cross validation")
    ax.legend()
    st.pyplot(fig)
    download_img(fig, fileName = "R2CV")


def plotPredictionCV(y,yhat,yhat_cv):
    report = regresssionReport(y,yhat)
    report_cv = regresssionReport(y,yhat_cv)
    fig, ax = plt.subplots()
    ax.plot([np.min(y)*0.95,np.max(y)*1.05],[np.min(y)*0.95,np.max(y)*1.05],
            color = 'black',label = "y=x")
    ax.scatter(y, yhat,color = 'tab:blue', marker='*', label ='Calibration')
    ax.scatter(y, yhat_cv,color = 'tab:orange', marker='.', label ='Cross Validation')
    ax.text(0.7, 0.05,
            "RMSEC = {:.2}\nR$^2$ = {:.2}\n\nRMSECV = {:.2}\nR$^2$$_c$$_v$ = {:.2}"\
                .format(report["rmse"],report["r2"],report_cv["rmse"],report_cv["r2"]),
            transform = ax.transAxes)
    ax.set_xlabel("Reference values")
    ax.set_ylabel("Prediction")
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Cross validation results")
    st.pyplot(fig)
    download_img(fig, fileName = "PredictionCV")
    
def plotPrediction(y,yhat, title = "Prediction results"):
    report = regresssionReport(y,yhat)
    fig, ax = plt.subplots()
    ax.plot([np.min(y)*0.95,np.max(y)*1.05],[np.min(y)*0.95,np.max(y)*1.05],
            color = 'black',label = "y=x")
    ax.scatter(y, yhat,color = 'tab:green', marker='*', label ='Prediction')
    ax.text(0.7, 0.03,
            "RMSEP = {:.2}\nR$^2$ = {:.2}".format(report["rmse"],report["r2"]),
            transform = ax.transAxes)
    ax.set_xlabel("Reference values")
    ax.set_ylabel("Prediction")
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.set_title(title)
    st.pyplot(fig)
    download_img(fig, fileName = "Prediction")

def plotRegressionCoefficients(b,wv=None):
    if wv==None:
        wv = np.linspace(1000,2500,len(b))
    fig, ax = plt.subplots()
    ax.plot(wv,b)
    ax.set_xlabel("Wavenumber (nm)")
    ax.set_ylabel("Coefficients")
    ax.set_title("Regression coefficients")
    st.pyplot(fig)
    download_img(fig, fileName = "RegressionCoefficients")


## plot for classfication by plsda
# plot rmsecv variation against lv
def plotAccuracyCV(accuracy_cv,labels = 'accuracy'):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(accuracy_cv))+1,accuracy_cv,color = 'tab:orange', 
            marker='.', label =labels)
    #ax.plot(optLV, r2cv[optLV-1],marker='o',markersize=10,label='optimal LV')
    ax.set_xlabel("Number of pls components")
    ax.set_ylabel(labels)
    ax.set_title("Variation of "+labels+" with nLV in calibration and cross validation")
    ax.legend()
    st.pyplot(fig)
    download_img(fig, fileName = "AccuracyCV")

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
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

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

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

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    st.pyplot(fig)
    download_img(fig, fileName = title)

    
