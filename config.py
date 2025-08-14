import importlib
import os

# EMBEDDING_TYPE = ['USNID', 'UnsupUSNID']
# DATASET_NAME = ['BANKING', 'CLINC', 'DBPEDIA', 'MCID', 'SNIPS', 'STACKOVERFLOW']

EMBEDDING_TYPE = 'USNID'
DATASET_NAME = 'CLINC'

# BASE CONFIG
base_config = {
    "data_list": [DATASET_NAME],
    "emb_type": [EMBEDDING_TYPE],
    "force_context_length": 10,
}


parameter_grids = {}
config_module_name = f"configs.{EMBEDDING_TYPE}_{DATASET_NAME}"

try:
    config_module = importlib.import_module(config_module_name)
    
    loaded_params = getattr(config_module, 'config', {})
    
    parameter_grids.update(loaded_params)
    
    if 'seeds' in base_config and 'seeds' not in parameter_grids:
        parameter_grids['seeds'] = base_config['seeds']

    print(f"Successfully loaded configuration from: {config_module_name}.py")

except ImportError:
    print(f"[ERROR] Could not find the configuration file: '{config_module_name}.py'")
    print("[ERROR] Please make sure the file exists in the 'configs' directory.")
    parameter_grids = {
        "seeds": base_config.get("seeds", [0]) 
    }
except Exception as e:
    print(f"[ERROR] An error occurred while loading the configuration: {e}")
    parameter_grids = {}


print(f"Running experiment for: {DATASET_NAME} with {EMBEDDING_TYPE} embeddings.")
print("Loaded parameters:", parameter_grids)