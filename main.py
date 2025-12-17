from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import shutil
import os
import uuid
import uvicorn

app = FastAPI()

# Підключаємо папку static для збереження тимчасових файлів
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- АЛГОРИТМ ---
def enhance_details_algorithm(image, radius, eps, boost):
    img_float = np.float32(image) / 255.0
    base_layer = cv2.ximgproc.guidedFilter(guide=img_float, src=img_float, radius=radius, eps=eps)
    detail_layer = img_float - base_layer
    enhanced = base_layer + (boost * detail_layer)
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

# --- МАРШРУТИ ---

@app.get("/")
def read_index():
    # Повертаємо наш красивий інтерфейс
    return FileResponse("index.html")

@app.post("/enhance/")
async def enhance_image(
    file: UploadFile = File(...),
    radius: int = Form(...), 
    eps: float = Form(...), 
    boost: float = Form(...) 
):
    unique_filename = f"{uuid.uuid4()}.png" # Безпечне ім'я
    input_path = f"static/input_{unique_filename}"
    output_path = f"static/result_{unique_filename}"
    
    # Збереження оригіналу
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    img = cv2.imread(input_path)
    if img is None: return {"error": "Error"}

    # Обробка
    result = enhance_details_algorithm(img, radius, eps, boost)
    
    # Збереження результату
    cv2.imwrite(output_path, result)
    
    # Повертаємо шлях до картинки, щоб JS міг її показати
    return {"url": output_path, "original_url": input_path}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)