FROM python:3.8

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install torch==1.12.1 torchvision==0.13.1

# --------------------------------------------------------------------------------------------
# Other packages
RUN pip install scikit-learn==1.1.2
RUN pip install https://github.com/waliens/multitask-dipath/archive/refs/heads/master.zip
RUN pip install https://github.com/Cytomine-ULiege/Cytomine-python-client/releases/download/v2.8.3/Cytomine-Python-Client-2.8.3.zip

# --------------------------------------------------------------------------------------------
# Download weights

RUN mkdir /app
RUN mkdir /root/.torch && mkdir /root/.torch/models
RUN mkdir /root/.cache/torch && mkdir /root/.cache/torch/hub/ && mkdir /root/.cache/torch/hub/checkpoints/
RUN wget https://download.pytorch.org/models/densenet121-a639ec97.pth -o /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
RUN wget https://download.pytorch.org/models/resnet50-0676ba61.pth -o /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
RUN wget https://dox.uliege.be/index.php/s/G72InP4xmJvOrVp/download -o /root/.torch/models/densenet121-mh-best-191205-141200.pth
RUN wget https://dox.uliege.be/index.php/s/kvABLtVuMxW8iJy/download -o /root/.torch/models/resnet50-mh-best-191205-141200.pth

# --------------------------------------------------------------------------------------------
# Files
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
