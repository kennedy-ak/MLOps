# Use the base image provided in the exercise
FROM python:3.12.3-slim

# Update pip and install pipenv
RUN pip install -U pip
RUN pip install pipenv


WORKDIR /app


COPY ["./Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY homework_1.py .
COPY models/model.bin ./models/


COPY ["homework_1.py", "models/model.bin", "./"]

#
ENTRYPOINT ["python", "homework_1.py"]
