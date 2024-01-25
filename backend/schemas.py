from sqlalchemy import Column, Integer, String, Float, Text
from backend.database import Base
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json

SIZE = 256

class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = False

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

class TextPickleType(TypeDecorator):

    impl = Text(SIZE)

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value

# Train Job Table
class Job_Train(Base):
    __tablename__ = "Job_Train"
    uid = Column(String(100), primary_key=True)
    type = Column(String(10), default=None)
    status = Column(String(10), default="In progress")
    params = Column(TextPickleType(), default=None)
    accuracy = Column(Float, default=None)
    run_time = Column(Float, default=None)

    def __repr__(self):
        return '<Job Train %r>' % (self.uid)


# Grid Search Job Table
class Job_GridSearch(Base):
    __tablename__ = "Job_GridSearch"
    uid = Column(String(100), primary_key=True)
    type = Column(String(10), default=None)
    status = Column(String(10), default="In progress")
    params = Column(TextPickleType(), default=None)
    best_accuracy = Column(Float, default=None)
    best_params = Column(TextPickleType(), default=None)
    run_time = Column(Float, default=None)

    def __repr__(self):
        return '<Job GridSearch %r>' % (self.uid)  