docker compose up -d

sleep 2 #try 2 seconds if getting connection refused error
aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration 

# pip install pipenv
python integration_test.py
ERROR_CODE=$?
if [ $ERROR_CODE -ne 0 ]; then
    docker compose logs
    docker compose down
    exit $ERROR_CODE
fi
python batch.py 2023 1
ERROR_CODE=$?
if [ $ERROR_CODE -ne 0 ]; then
    docker compose logs
    docker compose down
    exit $ERROR_CODE
fi

docker compose logs
docker compose down