from app.model.VO.UserVO import UserVO
from app.util_db.connection import connect


def newUser(name: str, email: str):
    cursor = connect.cursor()
    query = f"INSERT INTO users (name, email) values ('{name}', '{email}') RETURNING id"
    cursor.execute(query)
    connect.commit()
    id = cursor.fetchone()
    cursor.close()
    return id[0]


def getAllUsers():
    cursor = connect.cursor()
    query = "SELECT * FROM users"
    cursor.execute(query)
    users = []
    data_manager = cursor.fetchone()
    while data_manager:
        users.append(UserVO(data_manager[0], data_manager[1], data_manager[2]))
        data_manager = cursor.fetchone()
    cursor.close()
    return users


def getUserById(id: int):
    cursor = connect.cursor()
    query = f"SELECT * FROM users where id={id}"
    cursor.execute(query)
    data_manager = cursor.fetchone()
    if data_manager:
        cursor.close()
        return UserVO(data_manager[0], data_manager[1], data_manager[2])
    cursor.close()
    return None


def updateUser(newUser):
    cursor = connect.cursor()
    query = f"UPDATE users SET name = '{newUser.getName()}', email = '{newUser.getEmail()}' where id={newUser.getId()}"
    cursor.execute(query)
    connect.commit()
    cursor.close()


def deleteUser(id: int):
    cursor = connect.cursor()
    query = f"DELETE FROM users WHERE id={id}"
    cursor.execute(query)
    connect.commit()
    cursor.close()

