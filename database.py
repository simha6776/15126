# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DB_URL = "sqlite:///logistics.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String, unique=True, nullable=False),
    Column("email", String, unique=True, nullable=False),
    Column("password", String, nullable=False),
)

events = Table('events', metadata,
    Column('id', Integer, primary_key=True),
    Column('object_id', String),
    Column('object_type', String),
    Column('plate', String, nullable=True),
    Column('camera_id', String, nullable=True),
    Column('location', String, nullable=True),
    Column('timestamp', DateTime, default=datetime.utcnow),
    Column('frame', Integer),
    Column('x', Float),
    Column('y', Float),
    Column('w', Float),
    Column('h', Float),
)

shipments = Table('shipments', metadata,
    Column('id', Integer, primary_key=True),
    Column('shipment_code', String),
    Column('origin', String),
    Column('destination', String),
    Column('status', String),
    Column('assigned_plate', String, nullable=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)

stops = Table('stops', metadata,
    Column('id', Integer, primary_key=True),
    Column('shipment_id', Integer),
    Column('name', String),
    Column('lat', Float),
    Column('lon', Float),
    Column('sequence', Integer)
)

inventory = Table('inventory', metadata,
    Column('id', Integer, primary_key=True),
    Column('sku', String),
    Column('name', String),
    Column('qty', Integer),
    Column('location', String)
)

maintenance = Table('maintenance', metadata,
    Column('id', Integer, primary_key=True),
    Column('vehicle_id', String),
    Column('hours_running', Float),
    Column('miles', Float),
    Column('vibration', Float),
    Column('temp', Float),
    Column('rul', Float),
    Column('created_at', DateTime, default=datetime.utcnow)
)
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
