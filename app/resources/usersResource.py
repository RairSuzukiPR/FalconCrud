import falcon
from app.model.DAO import UserDAO
from falcon.media.validators.jsonschema import validate
from app.model.VO.UserVO import UserVO
from app.schemas import load_schema


class UsersListResource:
    @validate(resp_schema=load_schema('user/get/userRespSchema'))
    def on_get(self, req, resp):
        users = UserDAO.getAllUsers()
        users = [user.get_json() for user in users]

        resp.status = falcon.HTTP_200
        resp.media = users

    @validate(req_schema=load_schema('user/post/userReqSchema'))
    def on_post(self, req, resp):
        obj = req.get_media()
        newId = UserDAO.newUser(obj.get('name'), obj.get('email'))

        resp.status = falcon.HTTP_201
        resp.media = {
            'id': newId
        }

    @validate(req_schema=load_schema('user/put/userReqSchema'))
    def on_put(self, req, resp):
        obj = req.get_media()
        newUser = UserVO(obj.get('id'), obj.get('name'), obj.get('email'))
        UserDAO.updateUser(newUser)

        resp.status = falcon.HTTP_200
        resp.media = {
            'id': newUser.getId()
        }


class UsersResource:
    def on_get(self, req, resp, user_id):
        user = UserDAO.getUserById(user_id)
        if user:
            resp.media = user.get_json()
        else:
            resp.media = {"Message": "User doesnt exist"}

    def on_delete(self, req, resp, user_id):
        if UserDAO.getUserById(user_id):
            UserDAO.deleteUser(user_id)
            resp.media = {"Message": "User deleted"}
        else:
            resp.media = {"Message": "User doesnt exist"}


usersListResource = UsersListResource()
usersResource = UsersResource()
