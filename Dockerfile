FROM cytomineuliege/software-python3-base:latest
RUN pip install scikit-learn numpy tensorflow keras h5py hdf5 pillow
RUN mkdir -p /app
ADD run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
