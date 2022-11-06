import streamlit as st

st.markdown('''               
<style>
  .button {
    border: 1px solid #000000;
    color: #336699;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 32px;
    margin: 4px 2px;
    cursor: pointer;
}
  .largeFrame {
    border: 2px solid #000000;
    padding: 15px 32px;
    text-align: left;
    text-decoration: none;
    display: inline-block;
    margin: 4px 2px;
}
  .smallFrame {
    border: 2px solid #000000;
    padding: 15px 32px;
    text-align: left;
    text-decoration: none;
    display: inline-block;
    margin: 4px 2px;
}
  
</style>
<div class = "largeFrame">
    <a href="./Regression" class="button">Regression</a>
    <div class="smllFrame">
      <p>
        This is a user-frendily graphical module for training a regression model of NIR spectra. This module contains the functions of data upload, cross valudation and prediction of new spectra. 
      </p>
    </div>
</div>

<div class = "largeFrame">
    <a href="./Classification" class="button">Classification</a>
    <div class="smllFrame">
      <p>
         This is a user-frendily graphical module for training a classification model of NIR spectra. This module contains the functions of data upload, cross valudation and prediction of new spectra. 
      </p>
    </div>
</div>

<div class = "largeFrame">
    <a href="./Data_Preprocessing" class="button">Data Preprocessing</a>
    <div class="smllFrame">
      <p>
         This is a all-in-one, user-frendily graphical module for data preprocessing of NIR spectra. (In developing)
      </p>
    </div>
</div>

<div class = "largeFrame">
    <a href="./Outlier_Dection" class="button">Outlier Dection</a>
    <div class="smllFrame">
      <p>
         This is a all-in-one, user-frendily graphical module for outlier dection of NIR spectra. (In developing)
      </p>
    </div>
</div>

<div class = "largeFrame">
    <a href="./Calibration_Transfer" class="button">Calibration transfer</a>
    <div class="smllFrame">
      <p>
         This is a all-in-one, user-frendily graphical module for calibration transfer of NIR spectra. (In developing)
      </p>
    </div>
</div>

<div class = "largeFrame">
    <a href="./Utils" class="button">Utils</a>
    <div class="smllFrame">
      <p>
         This is a collection of functions of being useful for training a more accurate model. (In developing)
      </p>
    </div>    
</div>
            ''', unsafe_allow_html=True)
