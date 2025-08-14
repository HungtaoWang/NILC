import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_path = os.path.join(parent_dir, 'results')

print(f"========== Starting Per-Seed Results Processing from '{results_path}' ==========")

all_results_list = []

if not os.path.exists(results_path):
    print(f"Error: Results directory not found at '{results_path}'")
    exit()

all_result_files = [f for f in os.listdir(results_path) if f.endswith('.pkl')]

for file_name in sorted(all_result_files):
    file_path = os.path.join(results_path, file_name)
    try:
        with open(file_path, "rb") as f:
            results_dict_from_file = pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        print(f"  [Warning] Could not read corrupted file: {file_name}. Skipping.")
        continue

    try:
        base_name, _, file_suffix = file_name.removesuffix('.pkl').rpartition('_results_')
        emb_type, data_name = file_suffix.rsplit('_', 1)

        pattern = re.compile(
            r"Exp\d+_"  
            r"([a-zA-Z+]+)_"  
            r"i(\d+)_"  
            r"sb([\d.]+)_"  
            r"sg([\d.]+)_"  
            r"sd([\d.-]+)_"  
            r"(Semi\((.*?)\)|NoSemi)_"  
            r"(Opt_O(\d+)_nn(\d+)|NoOpt)"  
        )

        match = pattern.search(base_name)
        if not match:
            print(f"  [Warning] Filename '{file_name}' did not match expected pattern. Skipping.")
            continue

        groups = match.groups()
        rep_m = groups[0]
        max_i = groups[1]
        sb = groups[2]
        sg = groups[3]
        sd = groups[4]
        semi_full = groups[5] 
        opt_full = groups[7] 

        display_method_name = f"DNILC({rep_m},i={max_i},sb={sb},sg={sg},sd={sd},{semi_full}"

        if opt_full.startswith("Opt"):
            o_val = groups[8]
            nn_val = groups[9]
            display_method_name += f",Opt,O={o_val},nn={nn_val})"
        else:
            display_method_name += ",NoOpt)"

    except (ValueError, IndexError, AttributeError) as e:
        print(f"  [Warning] Could not parse filename: {file_name}. Error: {e}. Skipping.")
        continue

    seeds_found = list(results_dict_from_file.get(emb_type, {}).keys())
    if not seeds_found:
        continue

    for seed in seeds_found:
        seed_results = results_dict_from_file.get(emb_type, {}).get(seed, {})
        if not seed_results:
            continue

        for method_key, result_data in seed_results.items():
            if 'results' in result_data:
                results = result_data['results']
                
                final_method_name = method_key
                if method_key.startswith('DNILC'):
                    final_method_name = display_method_name

                all_results_list.append({
                    'Dataset': data_name,
                    'EmbeddingType': emb_type, 
                    'Method': final_method_name,
                    'Seed': seed,
                    'ACC': results[0] * 100,
                    'NMI': results[1] * 100,
                    'ARI': results[2] * 100,
                })
        
if all_results_list:
    df = pd.DataFrame(all_results_list)
    df.drop_duplicates(subset=['Dataset', 'EmbeddingType', 'Method', 'Seed'], inplace=True)

    for emb_type in sorted(df['EmbeddingType'].unique()):
        print(f"\n\n{'#'*30} RESULTS FOR EMBEDDING: {emb_type.upper()} {'#'*30}")
        df_emb_type = df[df['EmbeddingType'] == emb_type]

        for data_name in sorted(df_emb_type['Dataset'].unique()):
            print(f"\n\n{'='*25} DATASET: {data_name.upper()} {'='*25}")
            df_dataset = df_emb_type[df_emb_type['Dataset'] == data_name].copy()


            def get_method_type(method_name):
                if method_name in ['kmeans', 'kmedoids']: return 0
                if 'NoSemi' in method_name: return 1
                if 'Semi' in method_name: return 2
                return 3

            df_dataset['Method_Type'] = df_dataset['Method'].apply(get_method_type)
            
            df_dataset = df_dataset.sort_values(by=['Method_Type', 'ACC'], ascending=[True, False]).reset_index(drop=True)

            display_df = df_dataset[['Method', 'Seed', 'ACC', 'NMI', 'ARI']]
            
            pd.options.display.float_format = '{:,.2f}'.format
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', 200)

            print(display_df.to_string(index=False))
else:
    print("\nNo valid result files found to process in the 'results' directory.")