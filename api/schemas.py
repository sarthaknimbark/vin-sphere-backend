from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class AttributeConfidences(BaseModel):
    make: Optional[float] = None
    model: Optional[float] = None
    year: Optional[float] = None
    trim: Optional[float] = None
    body_type: Optional[float] = None
    cylinders: Optional[float] = None
    origin: Optional[float] = None
    no_of_passengers: Optional[float] = None
    weight: Optional[float] = None
    regional_spec: Optional[float] = None
    color: Optional[float] = None

class DecodeResponse(BaseModel):
    vin: str
    length: int
    make: str
    model: str
    year: str
    trim: str
    body_type: str
    cylinders: str
    origin: str
    no_of_passengers: str
    weight: str
    color: str
    regional_spec: str
    confidence: float
    attribute_confidences: AttributeConfidences
    source: str = "decoder" # "master_data" or "decoder"

class PriceRange(BaseModel):
    average: int
    minimum: int
    maximum: int

class ValuationDetail(BaseModel):
    retail_price: PriceRange
    trade_price: PriceRange

class ToyotaValuationWarning(BaseModel):
    message: str

class ToyotaValuationDetail(BaseModel):
    base_price: float
    depreciation_rate: float
    final_value: float
    adjustments: Dict[str, Any]
    warnings: list[str]

class APIResponse(BaseModel):
    decode_result: DecodeResponse
    lookup_valuation: Optional[ValuationDetail] = None
    ml_valuation: Optional[ValuationDetail] = None
    toyota_valuation: Optional[ValuationDetail] = None
    toyota_valuation_raw: Optional[ToyotaValuationDetail] = None
    depreciated_valuations: Dict[str, ValuationDetail] = Field(default_factory=dict)
