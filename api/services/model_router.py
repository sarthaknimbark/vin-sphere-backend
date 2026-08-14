import sys
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from api.config import MODEL_DIRS, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Cache dictionary to hold loaded modules
_predict_modules: Dict[int, Any] = {}

def load_all_predict_modules():
    """Preloads predict modules for all VIN/Chassis lengths to prevent runtime import latency."""
    # Ensure project root is in sys.path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
        
    for length, model_dir in MODEL_DIRS.items():
        if length == 17:
            # chat_cat(2.0) has a dot in its directory name, requiring spec loading
            predict_path = PROJECT_ROOT / "chat_cat(2.0)" / "predict.py"
            module_name = "chat_cat_2_predict"
        else:
            dir_name = f"chat_cat_short_vin_{length:02d}" if length in [9, 11, 12, 13] else "chat_cat_short_vin"
            predict_path = PROJECT_ROOT / dir_name / "predict.py"
            module_name = f"{dir_name}_predict"
            
        if not predict_path.exists():
            logger.warning(f"Predict module for length {length} not found at {predict_path}")
            continue
            
        try:
            # Set up the sys.path for individual short VIN imports to load relative imports properly
            dir_path = predict_path.parent
            if str(dir_path) not in sys.path:
                sys.path.append(str(dir_path))
                
            spec = importlib.util.spec_from_file_location(module_name, str(predict_path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            _predict_modules[length] = module
            logger.info(f"Successfully loaded predict module for length {length} as module '{module_name}'")
        except Exception as e:
            logger.error(f"Error loading predict module for length {length}: {e}", exc_info=True)

def decode_vin_via_model(vin: str, length: int) -> Dict[str, Any]:
    """Routes the VIN to the preloaded model for the given length and returns the prediction."""
    if length not in _predict_modules:
        raise ValueError(f"No predict module preloaded for VIN length {length}")
        
    module = _predict_modules[length]
    model_dir = str(MODEL_DIRS[length])
    
    if length == 17:
        # Standard VIN uses decode_vin
        res = module.decode_vin(vin, model_dir=model_dir)
    else:
        # Short VIN uses predict_vehicle
        res = module.predict_vehicle(vin, model_dir=model_dir)
        
    return res
