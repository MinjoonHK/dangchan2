from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.question import question_router
from domain.user import user_router

app = FastAPI()


#CORS 오류를 해결하기 위해 이렇게 할 수 있다.
# 프론트엔드의 URL을 통해서 할 수 있을듯!
# origins = [
#     'http://127.0.0.1:8000'
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credential=True,
#     allow_methods=['*'],
#     allow_headers=["*"]
# )

app.include_router(question_router.router)
app.include_router(user_router.router)