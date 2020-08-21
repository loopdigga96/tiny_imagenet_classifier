from sqlalchemy.orm import Session

from database import models, schemas


def get_all_requests(db: Session):
    return db.query(models.Requests).all()


def write_request(db: Session, request: schemas.Requests):
    db_request = models.Requests(**request.dict())
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    return db_request
