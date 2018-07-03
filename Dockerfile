FROM cytomineuliege/software-python3-base:latest
RUN pip install scikit-learn numpy tensorflow keras h5py hdf5 pillow
RUN mkdir -p /app
ADD classification_deep_features_model_builder_run.py /app/classification_deep_features_model_builder_run.py
ENTRYPOINT ["python", "/app/classification_deep_features_model_builder_run.py"]
