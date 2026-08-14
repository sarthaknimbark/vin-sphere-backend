import re
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional, Dict, Any

from api.schemas import APIResponse, DecodeResponse, ValuationDetail, AttributeConfidences, ToyotaValuationDetail
from api.services.model_router import decode_vin_via_model
from api.services.valuation_service import (
    get_lookup_valuation,
    get_ml_predicted_price_info,
    get_toyota_engine_valuation,
    get_depreciated_values_by_file,
    calculate_valuation
)
from api.config import get_db_connection_safe
from database import (
    lookup_vin_in_master_dataset,
    log_prediction,
    save_pending_prediction
)

router = APIRouter(prefix="/api/v1/decode", tags=["Decoding & Valuation"])

def format_csv_value(val) -> str:
    if val is None or pd.isna(val):
        return "UNKNOWN"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    return str(val)

def get_verified_vin_dict(row: dict) -> dict:
    attr_conf = {
        "make": 1.0, "model": 1.0, "year": 1.0, "trim": 1.0, "body_type": 1.0,
        "cylinders": 1.0, "origin": 1.0, "no_of_passengers": 1.0, "weight": 1.0,
        "regional_spec": 1.0, "color": 1.0
    }
    return {
        "vin": format_csv_value(row.get("chassisNumber")),
        "year": format_csv_value(row.get("year")),
        "make": format_csv_value(row.get("make")),
        "model": format_csv_value(row.get("model")),
        "trim": format_csv_value(row.get("trim")),
        "body_type": format_csv_value(row.get("bodyType")),
        "regional_spec": format_csv_value(row.get("regionalSpec")),
        "cylinders": format_csv_value(row.get("cylinders")),
        "color": format_csv_value(row.get("color")),
        "origin": format_csv_value(row.get("origin")),
        "weight": format_csv_value(row.get("weightInKg")),
        "no_of_passengers": format_csv_value(row.get("noOfPassengers")),
        "confidence": 1.0,
        "attribute_confidences": attr_conf
    }

def normalize_result_keys(result: dict) -> dict:
    mapped = {}
    mapped["make"] = str(result.get("make", "UNKNOWN")).strip().upper()
    mapped["model"] = str(result.get("model", "UNKNOWN")).strip().upper()
    mapped["year"] = str(result.get("year", "UNKNOWN")).strip().upper()
    mapped["trim"] = str(result.get("trim", "UNKNOWN")).strip().upper()
    mapped["body_type"] = str(result.get("body_type", "UNKNOWN")).strip().upper()
    mapped["cylinders"] = str(result.get("cylinders", "UNKNOWN")).strip().upper()
    mapped["origin"] = str(result.get("origin", "UNKNOWN")).strip().upper()
    mapped["no_of_passengers"] = str(result.get("no_of_passengers", "UNKNOWN")).strip().upper()
    mapped["weight"] = str(result.get("weight", "UNKNOWN")).strip().upper()
    mapped["color"] = str(result.get("color", "UNKNOWN")).strip().upper()

    reg_spec = result.get("regional_spec", result.get("regional_specs", "UNKNOWN"))
    mapped["regional_spec"] = str(reg_spec).strip().upper()

    # Normalize attribute confidences to ensure all keys are present
    raw_conf = result.get("attribute_confidences", result.get("confidence_scores", {}))
    normalized_conf = {}
    for k in ["make", "model", "year", "trim", "body_type", "cylinders", "origin", "no_of_passengers", "weight", "regional_spec", "color"]:
        val = raw_conf.get(k, raw_conf.get(k + "s", 0.0))  # handles regional_specs naming
        normalized_conf[k] = round(float(val), 4) if val is not None else 0.0

    mapped["attribute_confidences"] = normalized_conf
    mapped["confidence"] = result.get("confidence", 0.0)
    return mapped

def validate_17_char_vin(vin: str) -> tuple:
    vin = vin.upper().strip()
    if len(vin) != 17:
        return False, "Length must be exactly 17 characters."
    
    prohibited = [c for c in ("I", "O", "Q") if c in vin]
    if prohibited:
        return False, f"VIN cannot contain prohibited letters: {', '.join(prohibited)}."
        
    trans_map = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
        'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9
    }
    
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    
    total_sum = 0
    for idx in range(17):
        char = vin[idx]
        if char.isdigit():
            val = int(char)
        elif char in trans_map:
            val = trans_map[char]
        else:
            return False, f"Invalid character '{char}' in VIN."
            
        total_sum += val * weights[idx]
        
    remainder = total_sum % 11
    expected_check = "X" if remainder == 10 else str(remainder)
    actual_check = vin[8]
    
    if expected_check != actual_check:
         return False, f"Check-digit mismatch at 9th position (calculated '{expected_check}', but found '{actual_check}')."
        
    return True, ""

def process_decode_request(normalized_vin: str, length: int, include_valuation: bool) -> APIResponse:
    # 1. Check PostgreSQL Database (master_data)
    db_conn, db_err = get_db_connection_safe()
    db_ok = db_conn is not None
    if db_ok:
        try:
            matched_row = lookup_vin_in_master_dataset(normalized_vin)
            if matched_row:
                res = get_verified_vin_dict(matched_row)
                log_prediction(normalized_vin, length, "master_data", res)
                db_conn.close()
                return build_api_response(normalized_vin, length, res, "master_data", include_valuation)
        except Exception as e:
            print(f"Error querying master database: {e}")
        finally:
            try:
                db_conn.close()
            except Exception:
                pass

    # 2. Decode using the ML Model Router
    try:
        raw_res = decode_vin_via_model(normalized_vin, length)
        if raw_res is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Decoder model for length {length} returned empty result"
            )
        res = normalize_result_keys(raw_res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decoder failed: {str(e)}"
        )

    # 3. Log prediction and save to pending queue in DB
    if db_ok:
        try:
            db_conn, _ = get_db_connection_safe()
            log_prediction(normalized_vin, length, "decoder", res)
            save_pending_prediction(normalized_vin, length, res)
            db_conn.close()
        except Exception as e:
            print(f"Error writing to Postgres logs: {e}")
            try:
                db_conn.close()
            except Exception:
                pass

    return build_api_response(normalized_vin, length, res, "decoder", include_valuation)

def build_api_response(vin: str, length: int, decode_res: dict, source: str, include_valuation: bool) -> APIResponse:
    decode_response = DecodeResponse(
        vin=vin,
        length=length,
        make=decode_res.get("make", "UNKNOWN"),
        model=decode_res.get("model", "UNKNOWN"),
        year=decode_res.get("year", "UNKNOWN"),
        trim=decode_res.get("trim", "UNKNOWN"),
        body_type=decode_res.get("body_type", "UNKNOWN"),
        cylinders=decode_res.get("cylinders", "UNKNOWN"),
        origin=decode_res.get("origin", "UNKNOWN"),
        no_of_passengers=decode_res.get("no_of_passengers", "UNKNOWN"),
        weight=decode_res.get("weight", "UNKNOWN"),
        color=decode_res.get("color", "UNKNOWN"),
        regional_spec=decode_res.get("regional_spec", "UNKNOWN"),
        confidence=decode_res.get("confidence", 0.0),
        attribute_confidences=AttributeConfidences(**decode_res.get("attribute_confidences", {})),
        source=source
    )

    response_payload = APIResponse(decode_result=decode_response)

    if include_valuation:
        # Fetch lookup valuation
        lookup_val = get_lookup_valuation(decode_res, vin)
        if lookup_val is not None:
            response_payload.lookup_valuation = ValuationDetail(**lookup_val)

        # Fetch CatBoost ML valuation
        ml_price = get_ml_predicted_price_info(decode_res)
        if ml_price is not None:
            response_payload.ml_valuation = ValuationDetail(**calculate_valuation(ml_price))

        # Fetch Toyota valuation engine
        toyota_raw = get_toyota_engine_valuation(decode_res)
        if toyota_raw:
            final_val = toyota_raw.get("final_value")
            if final_val:
                response_payload.toyota_valuation = ValuationDetail(**calculate_valuation(final_val))
            response_payload.toyota_valuation_raw = ToyotaValuationDetail(
                base_price=toyota_raw.get("base_price", 0.0),
                depreciation_rate=toyota_raw.get("depreciation_rate", 0.0),
                final_value=final_val or 0.0,
                adjustments=toyota_raw.get("adjustments", {}),
                warnings=toyota_raw.get("warnings", [])
            )

        # Fetch depreciation valuations
        dep_vals = get_depreciated_values_by_file(decode_res)
        if dep_vals:
            dep_payloads = {}
            for k, val in dep_vals.items():
                calc = calculate_valuation(val)
                if calc:
                    dep_payloads[k] = ValuationDetail(**calc)
            response_payload.depreciated_valuations = dep_payloads

    return response_payload

@router.get("/standard/{vin}", response_model=APIResponse)
def decode_standard_vin(vin: str, include_valuation: bool = Query(False, description="Compute vehicle valuations")):
    """Decode a standard 17-character VIN using the PyTorch Transformer + backoff decoder."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin.upper())
    
    if len(normalized) != 17:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid length. Standard VIN must be exactly 17 characters (received {len(normalized)})."
        )
        
    is_valid, err_msg = validate_17_char_vin(normalized)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format validation failed: {err_msg}"
        )
        
    return process_decode_request(normalized, 17, include_valuation)

@router.get("/short/{vin}", response_model=APIResponse)
def decode_short_vin(vin: str, include_valuation: bool = Query(False, description="Compute vehicle valuations")):
    """Decode a short 9, 10, 11, 12, or 13 character chassis/VIN number using length-specific pipeline estimators."""
    normalized = re.sub(r"[^A-Z0-9]", "", vin.upper())
    length = len(normalized)
    
    if length not in [9, 10, 11, 12, 13]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid short chassis length ({length}). Must be 9, 10, 11, 12, or 13 characters."
        )
        
    return process_decode_request(normalized, length, include_valuation)
