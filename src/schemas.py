from pydantic import BaseModel, Field

class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str          # "male" / "female"
    bmi: float = Field(..., ge=10, le=80)
    children: int = Field(..., ge=0, le=20)
    smoker: str       # "yes" / "no"
    region: str     