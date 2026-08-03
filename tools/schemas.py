from pydantic import BaseModel


class RetrievalArgs(BaseModel):
    query: str


class CalculatorArgs(BaseModel):
    expression: str


class WebSerchArgs(BaseModel):
    query: str
