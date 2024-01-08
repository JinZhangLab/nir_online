
# NIR Online Tools

This is a web-based tool for performing calirabtion (i.e., regression, classiciatoin), data preprocessing, outlier dection, calibration transfer, etc., on near-infrared (NIR) spectroscopy data.


## Usage

### Docker compose (recommand)
```bash
# Clone the repository
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online

# Start the server
docker-compose up -d
```

Access NIR Online Tools at <http://localhost:8501>


### Docker 

You can deploy the online tools with docker on your local machine by runing the following commands:

``` bash
# Clone the repository
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online

docker build -t nir_online .

# Start the server
docker run -p 8501:8501 nir_online
```

Access NIR Online Tools at <http://localhost:8501>


### Python environment

``` bash
# Clone the repository
git clone https://github.com/JinZhangLab/nir_online.git
cd ./nir_online/app

# Install the requirements
python3 -m pip install requirements.txt

# Start the server
streamlit run Index.py
```

Access NIR Online Tools at <http://localhost:8501>


## Demo Server

You can also use the nir online tool in our demostrative server at <https://nir.chemoinfolab.com>.



## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. You can also report bugs and suggest new features by opening an issue.

## License

This project is licensed under the Apache License - see the LICENSE file for details.
