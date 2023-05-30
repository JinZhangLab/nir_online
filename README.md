
# NIR Online Tools

This is a web-based tool for performing calirabtion (i.e., regression, classiciatoin), data preprocessing, outlier dection, calibration transfer, e.g., on near-infrared (NIR) spectroscopy data.

## Usage

You can use this tools online at <https://nir.chemoinfolab.com> or depoly by your self.

## Deployment

### Docker (recommand)

You can deploy the online tools with docker on your local machine by runing the following commands:

``` bash
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online

docker build -t nir_online .

docker run -p 8501:8501 nir_online
```

Then, you can use the nir online tool by opening it in your browser at <http://localhost:8501>

### Local machine

``` bash
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online/app

python3 -m pip install requirements.txt

streamlit run Index.py
```

Then, you can use the nir online tool by opening it in your browser at <http://localhost:8501>

### Streamlit clould

[Instructions for deploying on Streamlit Cloud go here.](https://streamlit.io/cloud)

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. You can also report bugs and suggest new features by opening an issue.

## License

This project is licensed under the Apache License - see the LICENSE file for details.
