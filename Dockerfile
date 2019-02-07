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

# --------------------------------------------------------------------------------------------
# Download weights
RUN mkdir /app
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth -o /app/resnet50-19c8e357.pth
RUN wget https://download.pytorch.org/models/densenet201-c1103571.pth -o /app/densenet201-c1103571.pth

# --------------------------------------------------------------------------------------------
# Files
ADD beheaded_networks.py /app/beheaded_networks.py
ADD dataset.py /app/dataset.py
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
