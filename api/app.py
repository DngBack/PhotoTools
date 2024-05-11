from fastapi import FastAPI
from .router import router as api_router

# web_server_url = "http://localhost:8000"

app = FastAPI()
app.include_router(api_router)
