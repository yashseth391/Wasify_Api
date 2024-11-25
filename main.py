from typing import Union
from PIL import Image
from fastapi import FastAPI ,File,UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io
import cv2
from ultralytics import YOLO
modelPath = r"D:\College Minor Project\WasificationML\runs\detect\train\weights\best.pt"
model = YOLO('yolov8n.pt') 
model = YOLO(modelPath)
app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Model Run Successfully"}


@app.post("/predict/")
async def classify_image(file: UploadFile=File(...)):
    try:
        content=await file.read()
        image=Image.open(io.BytesIO(content))
        image = image.convert("RGB")
        new_size = (image.width * 2, image.height * 2)
        image = image.resize(new_size)
        image.save(r"D:\College Minor Project\FastApi learn\image.jpg")
        results=model(image)
        for result in results:
            result.show()
            result.save(filename=r"D:\College Minor Project\FastApi learn\ans.jpg")
        img=cv2.imread(r"D:\College Minor Project\FastApi learn\ans.jpg")    
        return {"message": "Image saved successfully."}
    except Exception as e:
        return {"error":str(e)}
    
@app.get("/ans/")
def ans():
    return FileResponse(r"D:\College Minor Project\FastApi learn\ans.jpg")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)