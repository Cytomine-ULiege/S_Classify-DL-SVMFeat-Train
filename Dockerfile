FROM neubiaswg5/ml-keras-base:latest

RUN pip install pillow scikit-image

ADD keras_util.py /app/keras_util.py
ADD cytomine_util.py /app/cytomine_util.py
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
