import importlib.util
import sys
import os

notebook_path = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(notebook_path, "..")))

import json, pickle
from dnilc import get_embeddings
from experiment_utils import load_dataset

file_path = "../dnilc.py"
spec = importlib.util.spec_from_file_location("dnilc", file_path)
dnilc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dnilc_module)

file_path = "../experiment_utils.py"
spec2 = importlib.util.spec_from_file_location("experiment_utils", file_path)
experiment_utils = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(experiment_utils)

all_data_types = ['mcid', 'banking', 'clinc', 'stackoverflow', 'snips', 'dbpedia']
all_emb_types = ['USNID', 'UnsupUSNID']

for data in all_data_types:
    print(f"\nProcessing dataset: {data}")
    
    labels, documents, num_clusters, prompt, text_type, instructor_prompt = experiment_utils.load_dataset(data, opt=data[-1])

    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels = [label_map[item] for item in labels]

    for emb_type in all_emb_types:
        print(f"  Generating embeddings for type: {emb_type}")
        
        embeddings = {}
        embeddings[emb_type] = dnilc_module.get_embeddings(documents, emb_type=emb_type, instructor_prompt="", dataset=data)

        data_dict = {
            'data': data,
            'labels': labels,
            'num_clusters': num_clusters,
            'documents': documents,
            'embeddings': embeddings,
            'prompt': prompt,
            'text_type': text_type
        }

        output_filename = f"../processed_data/{emb_type}_data_{data}.pkl"
        
        with open(output_filename, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"  -> Saved data to {output_filename}")

print("\nAll datasets and embedding types have been processed.")