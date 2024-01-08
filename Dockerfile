FROM python:3.12

EXPOSE 8501

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt

RUN  pip3 install  -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Index.py", "--server.port=8501", "--server.address=0.0.0.0"]