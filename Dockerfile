FROM python:3.10

ADD main.py .

RUN pip install poetry

CMD ["python", "./main.py"]

