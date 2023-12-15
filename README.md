
# NIR Online Tools

This is a web-based tool for performing calirabtion (i.e., regression, classiciatoin), data preprocessing, outlier dection, calibration transfer, etc., on near-infrared (NIR) spectroscopy data.

## Usage

You can use this tool online at <https://nir.chemoinfolab.com> or this alternative site <https://nironline.streamlit.app>, or you can deploy it yourself.

## Usage

### Docker compose (recommand)
```bash
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online

docker-compose up -d
```

Then, you can use the nir online tool by opening it in your browser at <http://localhost:8501>


### Docker 

You can deploy the online tools with docker on your local machine by runing the following commands:

``` bash
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online

docker build -t nir_online .

docker run -p 8501:8501 nir_online
```

Then, you can use the nir online tool by opening it in your browser at <http://localhost:8501>


### Python environment

``` bash
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online/app

python3 -m pip install requirements.txt

streamlit run Index.py
```

Then, you can use the nir online tool by opening it in your browser at <http://localhost:8501>


### Use in our demostrative server

You can also use the nir online tool in our demostrative server at <https://nir.chemoinfolab.com>.



## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. You can also report bugs and suggest new features by opening an issue.

## License

This project is licensed under the Apache License - see the LICENSE file for details.
