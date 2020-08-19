from sqlalchemy.orm import Session

from database import models, schemas


def get_all_requests(db: Session):
    return db.query(models.Requests).all()


def write_request(db: Session, request: schemas.Requests):
    db_request = models.Requests(infer_time=request.infer_time,
                                 infer_result=request.infer_result,
                                 predicted_class=request.predicted_class,
                                 confidence=request.confidence,
                                 client_host=request.client_host)
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    return db_request
