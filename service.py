from modules import LightningTinyImageNetClassifier
from train_solution import get_backbone, batch_size, lr, test_transform
from database import crud, models, schemas
from database.database import engine, SessionLocal

from typing import List
from PIL import Image
import io
import sys
import datetime

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# create db sqlite connection
models.Base.metadata.create_all(bind=engine)

# load best model
loss_function = torch.nn.modules.loss.CrossEntropyLoss()
full_checkpoint_path = ('./results/ef_b4_label_smoothing_early_stop_2020-08-19_11:55:12/'
                       'checkpoints/epoch=18-val_loss=1.77.ckpt')
best_model = LightningTinyImageNetClassifier(get_backbone(), loss_function, batch_size, lr)
best_model.load_state_dict(torch.load(full_checkpoint_path, map_location=torch.device('cpu'))['state_dict'])

# start app
app = FastAPI()


# Define the Response
class Prediction(BaseModel):
    predicted_class: int
    confidence: float
    probas: List[float] = []


@app.post("/predict/", response_model=Prediction)
async def create_upload_files(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')
    try:
        start_time = datetime.datetime.now()
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        img = test_transform(pil_image).unsqueeze(dim=0)

        logits = best_model(img).squeeze()

        probas = torch.nn.functional.softmax(logits, dim=0)
        predicted_class = probas.argmax(dim=0).item()
        confidence = probas[predicted_class].item()

        client_host = request.client.host
        print(client_host)
        # in milliseconds
        elapsed_time = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        request = schemas.Requests(infer_time=elapsed_time,
                                   infer_result=probas.tolist(),
                                   predicted_class=predicted_class,
                                   confidence=confidence,
                                   client_host=client_host)
        crud.write_request(db, request)

        return {'predicted_class': predicted_class, 'confidence': confidence,
                'probas': probas.tolist()}
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_all/", response_model=List[schemas.Requests])
def get_all(db: Session = Depends(get_db)):
    res = crud.get_all_requests(db)
    return res
