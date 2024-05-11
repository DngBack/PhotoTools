from pydantic import BaseModel


class InpaintingInRequest(BaseModel):
    image_base64: str
    mask_base64: str
    prompt: str
    negative_prompt: str
    

class InpaintingInResponse(BaseModel):
    status_code: int
    message: str
    image_base64: str
    
    
class BgChangingInRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str


class BgChangingInResponse(BaseModel):
    status_code: int
    message: str
    image_base64: str
    
    
class RemoveBgInRequest(BaseModel):
    image_base64: str
    

class RemoveBgInResponse(BaseModel):
    status_code: int
    message: str
    image_base64: str
    
    
class ModelGenInRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str
    

class ModelGenInResponse(BaseModel):
    status_code: int
    message: str
    image_base64: str