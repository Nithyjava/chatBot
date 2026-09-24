from pydantic import BaseModel


class MongoDbSchema(BaseModel):
    user_id: int
    bio: str
    status: bool

class UserQuery(BaseModel):
    query:str