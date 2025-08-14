import os
import numpy as np
import pickle
import warnings
import itertools
from dnilc import DNILC, get_embeddings
from experiment_utils import load_dataset, cluster_metrics
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from config import parameter_grids, base_config 
import pandas as pd

warnings.filterwarnings("ignore")

def generate_combinations(grid):
    if not grid: return [{}]
    keys, values = grid.keys(), grid.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*values)]

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

param_combinations = generate_combinations(parameter_grids)
print(f"Generated {len(param_combinations)} unique DNILC parameter combinations for experiments.")

for i, param_combo in enumerate(param_combinations):
    experiment_counter = i + 1
    
    current_config = {**base_config, **param_combo}

    is_opt = current_config.get("query_optimization", False)
    use_semi = current_config.get("use_semi_supervision", False)
    
    if use_semi:
        seeding_str = "Seed" if current_config.get("use_initial_seeding") else "NoSeed"
        cost_str = "Cost" if current_config.get("use_cost_modification") else "NoCost"
        map_str = current_config.get("mapping_strategy", "N/A")
        semi_str = f"Semi({seeding_str}-{cost_str}-{map_str})"
    else:
        semi_str = "NoSemi"
    
    opt_str = "Opt" if is_opt else "NoOpt"
    rep_m = current_config.get('rep_method', 'N/A')
    max_i = current_config.get('max_llm_iter', 'N/A')
    sb = current_config.get('summary_beta', '-')
    sg = current_config.get('summary_gamma', '-')
    sd = current_config.get('summary_delta', '-') if use_semi and current_config.get("use_cost_modification") else '-'
    
    exp_name_base = f"Exp{experiment_counter}_{rep_m}_i{max_i}_sb{sb}_sg{sg}_sd{sd}_{semi_str}"

    if is_opt:
        o_val = current_config.get('O', '-')
        nn_val = current_config.get('num_neighbor_clusters', '-')
        exp_name = f"{exp_name_base}_{opt_str}_O{o_val}_nn{nn_val}"
    else:
        exp_name = f"{exp_name_base}_{opt_str}"

    print(f"\n========== [{experiment_counter}/{len(param_combinations)}] Running Experiment: {exp_name} ==========")


    for data in current_config["data_list"]:
        print(f"\n----- Dataset: {data} -----")
        
        for emb_type in current_config["emb_type"]:
            print(f"--- Embedding Type: {emb_type} ---")

            if emb_type == 'UnsupUSNID':
                data_path = os.path.join('processed_data', f'UnsupUSNID_data_{data}.pkl')
            else:
                data_path = os.path.join('processed_data', f'USNID_data_{data}.pkl')

            try:
                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                print(f"  [Error] Data file not found: {data_path}. Skipping.")
                continue
        
        labels = data_dict['labels']
        num_clusters = data_dict['num_clusters']
        documents = data_dict['documents']
        text_features = data_dict['embeddings']
        prompt = data_dict['prompt']
        text_type = data_dict['text_type']

        known_centroids = None
        known_labels_list = None
        if current_config.get("use_semi_supervision", False):
            print("  [Semi-Supervision] Loading signals...")
            try:
                semi_signals_dir = 'semi_signals'
                known_labels_file = os.path.join(semi_signals_dir, f'USNID_{data}_0.75_known_labels.tsv')
                labeled_samples_file = os.path.join(semi_signals_dir, f'USNID_{data}_0.75_0.1_labeled_training_samples.tsv')

                known_labels_df = pd.read_csv(known_labels_file, sep='\t', header=0)
                known_labels_list = known_labels_df['known_label'].tolist() 
                print(f"  [Semi-Supervision] Loaded {len(known_labels_list)} known intent labels.")

                samples_df = pd.read_csv(labeled_samples_file, sep='\t', header=0)

                samples_by_label = samples_df.groupby('label')['text'].apply(list).to_dict()
                
                known_centroids_map = {}
                for label, texts in samples_by_label.items():
                    if label in known_labels_list:
                        sample_embeddings = get_embeddings(texts, emb_type=current_config["emb_type"][0], dataset=data)
                        if sample_embeddings.shape[0] > 0:
                            known_centroids_map[label] = np.mean(sample_embeddings, axis=0)
                
                known_centroids = np.array([known_centroids_map[label] for label in known_labels_list if label in known_centroids_map])
                print(f"  [Semi-Supervision] Calculated {known_centroids.shape[0]} seed centroids.")

            except Exception as e:
                print(f"  [Error] Failed to process semi-supervision signals: {e}. Skipping.")
                known_centroids = None
                known_labels_list = None

        for emb_type in current_config["emb_type"]:
            print(f"--- Embedding Type: {emb_type} ---")
            results_dict = {emb_type: {}}
            emb_data = text_features[emb_type]

            oracle_clustered_embeddings = {i: [] for i in range(num_clusters)}
            for embedding, label in zip(emb_data, labels):
                if label < num_clusters:
                    oracle_clustered_embeddings[label].append(embedding)
            oracle_centroids = [np.mean(oracle_clustered_embeddings[i], axis=0) if oracle_clustered_embeddings[i] else np.zeros(emb_data.shape[1]) for i in range(num_clusters)]
            oracle_summary_embeddings = np.array(oracle_centroids)

            for seed in current_config["seeds"]:
                print(f"- Running Seed: {seed}")
                results_dict[emb_type][seed] = {}

                kmeans = KMeans(n_clusters=num_clusters, max_iter=120, n_init=1, random_state=seed)
                kmeans_assignments = kmeans.fit_predict(emb_data)
                results_k = cluster_metrics(np.array(labels), kmeans_assignments, kmeans.cluster_centers_, oracle_centroids, oracle_summary_embeddings)
                results_dict[emb_type][seed]['kmeans'] = {'results': results_k}

                kmedoids = KMedoids(n_clusters=num_clusters, max_iter=120, random_state=seed)
                kmedoids_assignments = kmedoids.fit_predict(emb_data)
                results_km = cluster_metrics(np.array(labels), kmedoids_assignments, emb_data[kmedoids.medoid_indices_], oracle_centroids, oracle_summary_embeddings)
                results_dict[emb_type][seed]['kmedoids'] = {'results': results_km}

                dnilc_params = {
                    **current_config,
                    "text_data": documents,
                    "num_clusters": num_clusters,
                    "prompt": prompt,
                    "text_type": text_type,
                    "emb_type": emb_type,
                    "random_state": seed,
                    "text_features": emb_data,
                    "data": data,
                    "true_labels": labels,
                    "known_centroids": known_centroids,
                    "known_labels": known_labels_list
                }
                
                valid_dnilc_keys = DNILC.__code__.co_varnames
                filtered_dnilc_params = {k: v for k, v in dnilc_params.items() if k in valid_dnilc_keys}

                assignments, summaries, summary_embeds, final_centroids, _, _, query_log = DNILC(**filtered_dnilc_params)
                
                results = cluster_metrics(np.array(labels), assignments, final_centroids, oracle_centroids, oracle_summary_embeddings, summary_embeds)
                
                method_key = f"DNILC_{exp_name}"
                results_dict[emb_type][seed][method_key] = {'results': results, 'params': filtered_dnilc_params, 'query_log': query_log}
                print(f"    -> Results for {method_key}: ACC={results[0]:.4f}, NMI={results[1]:.4f}, ARI={results[2]:.4f}")

            output_filename = os.path.join(results_dir, f'{exp_name}_results_{emb_type}_{data}.pkl')
            with open(output_filename, "wb") as f:
                pickle.dump(results_dict, f)
            print(f"--> Results for experiment '{exp_name}' on dataset '{data}' saved to {output_filename}")