FROM python:3.9

WORKDIR /app

RUN pip install flask
RUN pip install seaborn
RUN pip install pymorphy2
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install regex
RUN pip install fire
RUN pip install redis
RUN pip install celery
RUN pip install pymongo
RUN pip install "pymongo[srv]"
RUN pip install nltk
RUN pip install openTSNE

COPY ./tools/app ./tools/app
COPY ./substwsi ./substwsi
COPY ./run_tools.py ./run_tools.py

RUN chmod +x ./tools/app/start-worker.sh

ENTRYPOINT ["python3", "run_tools.py"]

EXPOSE 8080