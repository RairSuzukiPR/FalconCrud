import falcon
from app.resources import usersResource
from app.util_db import tables


app = falcon.App()

app.add_route('/users', usersResource.usersListResource)
app.add_route('/users/{user_id}', usersResource.usersResource)



