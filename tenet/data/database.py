import contextlib
from typing import Union, Dict, Callable, Any

from sqlalchemy import Column, Integer, String, ForeignKey, inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, relationship
from sqlalchemy_utils import create_database, database_exists

from tenet.core.exc import TenetError

Base = declarative_base()

'''
class Instance(Base):
    __tablename__ = "instance"

    id = Column('id', String, primary_key=True)
    name = Column('name', String, nullable=False)
    image = Column('image', String, nullable=False)
    volume = Column('volume', String, nullable=False)
    status = Column('status', String, nullable=False)
    kind = Column('kind', String, nullable=False)
    ip = Column('ip', String, nullable=False)
    port = Column('port', Integer, nullable=False)

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'image': self.image, 'volume': self.volume, 'status': self.status,
                'kind': self.kind, 'ip': self.ip, 'port': self.port}

    def __str__(self):
        return str(self.to_dict())
'''

'''
class NexusData(Base):
    __tablename__ = "nexus"
    id = Column('id', Integer, primary_key=True)
    name = Column('name', String, nullable=False)
    tid = Column('tid', String, ForeignKey('container.id'), nullable=False)
    bid = Column('bid', String, ForeignKey('container.id'), nullable=False)
    status = Column('status', String, nullable=False)
    tool = relationship("ContainerData", foreign_keys=[tid])
    benchmark = relationship("ContainerData", foreign_keys=[bid])
'''


class Database:
    def __init__(self, dialect: str, username: str, password: str, host: str, port: int, database: str,
                 debug: bool = False):

        self.url = f"{dialect}://{username}:{password}@{host}:{port}/{database}"

        if not database_exists(self.url):
            try:
                create_database(url=self.url, encoding='utf8')
            except TypeError as te:
                raise TenetError(f"Could not create database @{host}:{port}/{database}. {te}")

        self.engine = create_engine(self.url, echo=debug)
        Base.metadata.create_all(bind=self.engine)

    def refresh(self, entity: Base):
        with Session(self.engine) as session, session.begin():
            session.refresh(entity)

        return entity

    def add(self, entity: Base):
        with Session(self.engine) as session, session.begin():
            session.add(entity)
            session.flush()
            session.refresh(entity)
            session.expunge_all()

            if hasattr(entity, 'id'):
                return entity.id

    def destroy(self):
        # metadata = MetaData(self.engine, reflect=True)
        with contextlib.closing(self.engine.connect()) as con:
            trans = con.begin()
            Base.metadata.drop_all(bind=self.engine)
            trans.commit()

    def delete(self, entity: Base, entity_id: Union[int, str]):
        with Session(self.engine) as session, session.begin():
            return session.query(entity).filter(entity.id == entity_id).delete(synchronize_session='evaluate')

    def has_table(self, name: str):
        inspector = inspect(self.engine)
        return inspector.reflect_table(name, None)

    def filter(self, entity: Base, filters: Dict[Any, Callable], distinct: Any = None):
        with Session(self.engine) as session, session.begin():
            query = session.query(entity)

            for attr, exp in filters.items():
                query = query.filter(exp(attr))
            if distinct:
                query = query.distinct(distinct)
            session.expunge_all()
            return query

    def query(self, entity: Base, entity_id: Union[int, str] = None):
        with Session(self.engine) as session, session.begin():
            if entity_id and hasattr(entity, 'id'):
                results = session.query(entity).filter(entity.id == entity_id).first()
            else:
                results = session.query(entity).all()

            session.expunge_all()
            return results

    def query_attr(self, entity: Base, entity_id: int, attr: str):
        with Session(self.engine) as session, session.begin():
            if hasattr(entity, 'id') and hasattr(entity, attr):
                results = session.query(entity).filter(entity.id == entity_id).first()
                attr_result = getattr(results, attr)
                session.expunge_all()
                return attr_result

    def update(self, entity: Base, entity_id: int, attr: str, value):
        with Session(self.engine) as session, session.begin():
            if hasattr(entity, 'id') and hasattr(entity, attr):
                session.query(entity).filter(entity.id == entity_id).update({attr: value})
            else:
                raise TenetError(f"Could not update {type(entity)} {attr} with value {value}")
