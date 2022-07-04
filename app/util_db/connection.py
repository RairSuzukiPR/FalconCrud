import psycopg2


connect = psycopg2.connect(
    host='127.0.0.1',
    database='falconcrud',
    user='postgres',
    password='12345678'
)
