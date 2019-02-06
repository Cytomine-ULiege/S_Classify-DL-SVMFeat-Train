FROM python:3.6

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git
RUN cd /Cytomine-python-client && git checkout tags/v2.2.0 && pip install .
RUN rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision==0.2.1 pydensecrf==1.0rc3

# --------------------------------------------------------------------------------------------
# Other packages
RUN pip install scikit-learn==0.20

COPY beheaded_networks.py /app/beheaded_networks.py
COPY resnet50-19c8e357.pth /app/resnet50-19c8e357.pth
COPY densenet201-c1103571.pth /app/densenet201-c1103571.pth
COPY dataset.py /app/dataset.py
COPY run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
