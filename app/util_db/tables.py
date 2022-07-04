from app.util_db.connection import connect


def createTableUser():
    cursor = connect.cursor()
    table = 'CREATE TABLE IF NOT EXISTS users (id serial primary key, name varchar(45), email varchar(45))'
    cursor.execute(table)
    connect.commit()
    cursor.close()


createTableUser()
