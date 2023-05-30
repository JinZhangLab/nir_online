FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

COPY ./app /app

RUN pip3 install streamlit && \
    pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Index.py", "--server.port=8501", "--server.address=0.0.0.0"]