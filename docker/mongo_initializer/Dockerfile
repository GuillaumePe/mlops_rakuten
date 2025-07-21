FROM python:3.10-slim

RUN apt-get update && apt-get install -y netcat-openbsd

RUN pip install pymongo polars

COPY ./src/data /src
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

CMD bash -c "./wait-for-it.sh mongodb 27017 && echo Lancement scripts; python /src/upload_to_mongodb.py; python /src/reboot.py;  echo Fin scripts"
