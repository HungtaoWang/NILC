import json
import os

OLD_CACHE_FILE = "llm_cache.json"
NEW_CACHE_FILE = "new_cache.json" 

def load_cache(cache_file_path):
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache_data, cache_file_path):
    with open(cache_file_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)