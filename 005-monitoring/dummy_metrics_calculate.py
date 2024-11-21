# 



import datetime
import time
import random
import logging
import uuid
import pytz
import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics(
    timestamp TIMESTAMP,
    value1 INTEGER,
    value2 VARCHAR,
    value3 FLOAT
);
"""

def prep_db():
    # Ensuring autocommit is on for database creation and initial table setup
    with psycopg.connect("host=localhost port=5433 user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            if len(cur.fetchall()) == 0:
                cur.execute("CREATE DATABASE test;")
                logging.info("Database 'test' created successfully.")

    with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
        conn.execute(create_table_statement)
        logging.info("Table 'dummy_metrics' created successfully.")

def calculate_dummy_metrics_postgresql(curr):
    value1 = rand.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = rand.random()
    curr.execute(
        "INSERT INTO dummy_metrics(timestamp, value1, value2, value3) VALUES (%s, %s, %s, %s)",
        (datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
    )
    logging.info(f"Data inserted: {value1}, {value2}, {value3}")

def check_data():
    with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM dummy_metrics ORDER BY timestamp DESC LIMIT 10;")
            rows = cur.fetchall()
            for row in rows:
                print(row)

def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 100):
            with conn.cursor() as curr:
                calculate_dummy_metrics_postgresql(curr)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("Data sent")

if __name__ == '__main__':
    main()
    check_data()
