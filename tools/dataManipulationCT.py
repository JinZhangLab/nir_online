# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:44:30 2022

@author: chinn
"""
import streamlit as st
import os
from scipy.io import loadmat
basePath = os.path.split(os.path.realpath(__file__))[0]

@st.cache
def get_Tablet():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-2, ycal; Xtrans1-2, ytrans; Xtest1-2, ytest and wv

    """
    data = loadmat(basePath+"./Data/Tablet.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X":(data["Xcal1"],data["Xcal2"]),"y":data["ycal"]}
    dataOut["Trans"] = {"X":(data["Xtrans1"],data["Xtrans2"]),"y":data["ytrans"]}
    dataOut["Test"] = {"X":(data["Xtest1"],data["Xtest2"]),"y":data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut

@st.cache
def get_PlantLeaf():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-2, ycal; Xtrans1-2, ytrans; Xtest1-2, ytest and wv

    """
    data = loadmat(basePath+".\Data\PlantLeaf.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X":(data["Xcal1"],data["Xcal2"]),"y":data["ycal"]}
    dataOut["Trans"] = {"X":(data["Xtrans1"],data["Xtrans2"]),"y":data["ytrans"]}
    dataOut["Test"] = {"X":(data["Xtest1"],data["Xtest2"]),"y":data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut

@st.cache
def get_Corn():
    """
    Returns
    -------
    data : dict
        A dict containing Xcal1-3, ycal; Xtrans1-3, ytrans; Xtest1-3, ytest and wv

    """
    data = loadmat(basePath+"\Data\Corn.mat")
    dataOut = dict()
    dataOut["Cal"] = {"X":(data["Xcal1"],data["Xcal2"],data["Xcal3"]),"y":data["ycal"]}
    dataOut["Trans"] = {"X":(data["Xtrans1"],data["Xtrans2"],data["Xtrans3"]),"y":data["ytrans"]}
    dataOut["Test"] = {"X":(data["Xtest1"],data["Xtest2"],data["Xtest3"]),"y":data["ytest"]}
    dataOut["wv"] = data["wv"]
    return dataOut

    
if __name__ == "__main__":
    data = get_Tablet()
    print(data)