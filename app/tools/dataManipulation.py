# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:44:30 2022

@author: Jin Zhang
"""
import base64
import io
import streamlit as st
import pandas as pd
import time
import streamlit as st
import numpy as np

import os
from scipy.io import loadmat

basePath = os.path.split(os.path.realpath(__file__))[0]


# @st.cache_data
def convert_fig(fig, figFormat="pdf"):
    img = io.BytesIO()
    fig.savefig(img, format=figFormat)
    return img


def get_download_img_url(fig, figFormat="pdf", fileName="Figure"):
    # Convert the figure to bytes
    img = convert_fig(fig, figFormat=figFormat)
    # Encode the bytes to base64
    b64 = base64.b64encode(img.getvalue()).decode()
    # Create a markdown link to download the image
    link = f'<a href="data:image/{figFormat};base64,{b64}" download="{fileName}.{figFormat}">Download image</a>'
    return link

def download_img(fig, figFormat="pdf", fileName="Figure", label="Download image"):
    link = get_download_img_url(fig, figFormat=figFormat, fileName=fileName)
    # Display the link in streamlit
    st.markdown(link, unsafe_allow_html=True)




@st.cache_data
def get_download_csv_url(df, index=True, columns=True, index_label=None):
    # Convert the dataframe to csv
    csv = df.to_csv(index=index, header=columns, index_label=index_label)
    # Encode the csv to base64
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

def download_csv(df, index=True, columns=True, index_label=None, fileName="data", label="Download"):
    b64 = get_download_csv_url(df, index=index, columns=columns, index_label=index_label)
    # Create a markdown link to download the csv
    link = f'<a href="data:file/csv;base64,{b64}" download="{fileName + ".csv"}"> {label}</a>'
    # Display the link in streamlit
    st.markdown(link, unsafe_allow_html=True)


@st.cache_data
def get_Tablet():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-2, ycal; Xtrans1-2, ytrans; Xtest1-2, ytest and wv

    """
    data = loadmat(basePath+"/Data/Tablet.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X": (data["Xcal1"], data["Xcal2"]), "y": data["ycal"]}
    dataOut["Trans"] = {
        "X": (data["Xtrans1"], data["Xtrans2"]), "y": data["ytrans"]}
    dataOut["Test"] = {
        "X": (data["Xtest1"], data["Xtest2"]), "y": data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut


@st.cache_data
def get_PlantLeaf():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-2, ycal; Xtrans1-2, ytrans; Xtest1-2, ytest and wv

    """
    data = loadmat(basePath+"/Data/PlantLeaf.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X": (data["Xcal1"], data["Xcal2"]), "y": data["ycal"]}
    dataOut["Trans"] = {"X": (data["Xtrans1"], data["Xtrans2"]), "y": data["ytrans"]}
    dataOut["Test"] = {"X": (data["Xtest1"], data["Xtest2"]), "y": data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut


@st.cache_data
def get_Corn():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-3, ycal; Xtrans1-3, ytrans; Xtest1-3, ytest and wv

    """
    data = loadmat(basePath+"/Data/Corn.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X": (data["Xcal1"], data["Xcal2"],
                            data["Xcal3"]), "y": data["ycal"]}
    dataOut["Trans"] = {
        "X": (data["Xtrans1"], data["Xtrans2"], data["Xtrans3"]), "y": data["ytrans"]}
    dataOut["Test"] = {
        "X": (data["Xtest1"], data["Xtest2"], data["Xtest3"]), "y": data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut


def predict_reg(X, model):
    """
    Predicts the reference values for a set of spectra using a linear regression
    model.

    Args:
        X (pandas.DataFrame): A DataFrame containing the spectra to be
        predicted. Each row represents a sample and each column represents a
        wavelength. model (pandas.DataFrame): A DataFrame containing the
        coefficients of the linear regression model. The first element
        represents the intercept and each subsequent element represents the
        coefficient for a wavelength.

    Returns:
        pandas.DataFrame: A DataFrame containing the predicted reference values.
        Each row represents a sample and the only column represents the
        predicted reference value.

    Computes the predicted reference values for the input spectra using the
    linear regression model. The input spectra are augmented with a column of
    ones to account for the intercept term in the model. The dot product of the
    augmented spectra and the model coefficients is computed to obtain the
    predicted reference values. The resulting DataFrame is returned with the
    same index as the input spectra and a single column named "Prediction".
    """
    ones = np.ones((X.to_numpy().shape[0], 1))
    X_aug = np.hstack((ones, X.to_numpy()))
    yhat = np.dot(X_aug, np.reshape(model.to_numpy(), (-1, 1)))
    return pd.DataFrame(yhat, index=X.index.to_numpy(), columns=["Prediction"])



if __name__ == "__main__":
    data = get_Tablet()
    print(data)
