FROM python:3.7-slim-buster
RUN pip3 install --upgrade pip
RUN pip3 install flask flask_cors pillow 
RUN pip3 install torch==1.4.0 torchvision==0.5.0
RUN mkdir app
COPY . /app
EXPOSE 8095
WORKDIR /app
ENTRYPOINT ["python3", "app.py"]