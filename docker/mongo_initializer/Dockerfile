FROM python:3.10-slim

RUN apt-get update && apt-get install -y netcat-openbsd

RUN pip install pymongo polars

COPY ./src/data /src
#script pour attendre que postegre se deploie
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

CMD ["./wait-for-it.sh", "mongodb", "27017", "--", "bash", "-c", "python /src/reboot.py && python /src/upload_to_mongodb.py"]