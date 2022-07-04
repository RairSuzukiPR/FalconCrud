class UserVO:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

    def get_json(self):
        return dict(
            id=self.id,
            nome=self.name,
            email=self.email
        )

    def getId(self):
        return self.id

    def getName(self):
        return self.name

    def getEmail(self):
        return self.email
