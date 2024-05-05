import uuid
from fastapi import FastAPI
from .logger import logger


app = FastAPI()


@app.get("/")
async def root():
    logger.info('Hello world')
    return {"message": "Hello World"}


@app.get("/error")
async def error(req_id: str = ""):
    if req_id == "":
        req_id = str(uuid.uuid4())
    extra_logging = {'req_id': req_id}
    for i in range(0, 40):
        if i % 2 == 0:
            logger.error('Had an issue', exc_info=True, extra=extra_logging)
        else:
            logger.info('Successful request', exc_info=True, extra=extra_logging)
    return {'message': 'Had an issue'}
