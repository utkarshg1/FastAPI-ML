from pydantic import BaseModel, Field
from typing import Literal, Dict


class IrisInference(BaseModel):
    sepal_length: float = Field(description="Length of sepal in cm", ge=0)
    sepal_width: float = Field(description="Width of sepal in cm", ge=0)
    petal_length: float = Field(description="Length of petal in cm", ge=0)
    petal_width: float = Field(description="Width of petal in cm", ge=0)


class IrisPrediction(BaseModel):
    prediction: Literal["setosa", "virginica", "versicolor"]
    probability: Dict[str, float]
