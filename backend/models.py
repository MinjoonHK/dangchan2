from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Date
from sqlalchemy.orm import relationship

from database import Base

# 유저가 보내는 질문 정보
class Question(Base):
    __tablename__ = 'question'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    user = relationship('User', backref='questions')
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    
# 유저의 정보
class User(Base):
    __tablename__ = 'user'
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True,nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, unique=True, nullable=False)
    birth_date = Column(Date, nullable=False)
    address = Column(String, nullable=False)
    # phone_number = Column(String, nullable=False)/
    

# 질문에 의한 답변 정보
class Answer(Base):
    __tablename__ = 'answer'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    question_id = Column(Integer, ForeignKey('question.id'))
    user = relationship('User', backref='answer')
    question = relationship('Question', backref="answer")
    