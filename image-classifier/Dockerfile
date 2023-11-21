FROM python:3.9.10
WORKDIR /image_classifier
COPY config.py server.py model.py requirements.txt ./
COPY model model/
COPY utils utils/
RUN pip install -r requirements.txt
CMD ["flask", "--app", "server", "run", "--host=0.0.0.0"]