# to run the code: fastapi dev api.py
from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # if file.content_type == "image/jpeg":
    #     return Exception("JPEG files not allowed")
    
    contents = await file.read()    # 동시 요청 가능하게 함
    return {"filename": file.filename, "filesize": len(contents)}
