import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch import nn
from easydict import EasyDict
from tqdm import tqdm
import json
import jsonlines
import math
import os
import time
import errno
import signal
import functools
import requests
import re 
from scipy.spatial.distance import cdist 
from scipy.optimize import linear_sum_assignment 
from cache_utils import load_cache, save_cache, OLD_CACHE_FILE, NEW_CACHE_FILE

import openai

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class BERT_USNID_UNSUP(BertPreTrainedModel):
    
    def __init__(self, config, args):

        super(BERT_USNID_UNSUP, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
 
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.mlp_head = nn.Linear(config.hidden_size, args.num_labels)
            
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, weights = None,
                feature_ext = False, mode = None, loss_fct = None, aug_feats=None, use_aug = False):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)
            
        features = self.dense(features)
        pooled_output = self.activation(features)   
        pooled_output = self.dropout(features)
        
        logits = self.classifier(pooled_output)
            
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            return features, mlp_outputs
        else:
            return mlp_outputs, logits


class BERT_USNID(BertPreTrainedModel):
    """
    This is the custom BERT model class from the textoir/backbones/bert.py file,
    necessary for loading the fine-tuned USNID model weights and architecture.
    """
    def __init__(self, config, args):
        super(BERT_USNID, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
        
        self.classifier = None
        self.mlp_head = None
        
        if hasattr(args, 'num_labels') and args.num_labels is not None:
                self.classifier = nn.Linear(config.hidden_size, args.num_labels)
                self.mlp_head = nn.Linear(config.hidden_size, args.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, feature_ext=False, **kwargs):
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        features = sum_embeddings / sum_mask
        
        pooled_output = self.dense(features)
        pooled_output = self.activation(pooled_output)      
        pooled_output = self.dropout(pooled_output)
        
        if feature_ext:
            return pooled_output, None  
        
        return pooled_output

def call_chatgpt(prompt, new_cache, old_cache, num_predictions=1, temperature=0, max_tokens=1000, timeout=5.0):
    if prompt in new_cache:
        return new_cache[prompt]

    if prompt in old_cache:
        result = old_cache[prompt]
        new_cache[prompt] = result
        print(f"  [Cache] Found in old cache, moved to new cache.")
        return result

    apiKey = ""
    basicUrl = ""
    modelName = ""
    apiVersion = ""
    url = f"{basicUrl}/deployments/{modelName}/chat/completions/?api-version={apiVersion}"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {
        'messages': [{"role": "user", "content": prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': num_predictions
    }
    
    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if content:
                    new_cache[prompt] = content
                
                return content
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception during API call: {e}")

@timeout(5, os.strerror(errno.ETIMEDOUT))
def call_embedding(text, timeout=2.0):
    apiKey = ""
    basicUrl = ""
    modelName = ""
    apiVersion = ""
    url = f"{basicUrl}/deployments/{modelName}/embeddings?api-version={apiVersion}"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {"input": text, "encoding_format": "float"}

    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                embeddings = response.json()['data'][0]['embedding']
                return embeddings
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception during API call: {e}")

def get_embeddings(texts, model="text-embedding-3-small", emb_type='openai', instructor_prompt="", dataset=None):
    if emb_type == 'openai':
        embeddings = [call_embedding(text) for text in texts]
        return np.array([e for e in embeddings if e is not None])

    elif emb_type == 'distilbert':
        model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        embeddings = model.encode(texts)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.array(normalized_embeddings)

    elif emb_type.lower() == 'unsupusnid':
        print("Getting embeddings using fine-tuned UnsupUSNID model...")
        if dataset is None:
            raise ValueError("Parameter 'dataset' must be provided when emb_type is 'unsupusnid'.")
        
        usnid_paths = {
            "stackoverflow": "models/UnsupUSNID/UnsupUSNID_stackoverflow_1.0_bert_USNID_Unsup_9/models",
            "clinc": "models/UnsupUSNID/UnsupUSNID_clinc_1.0_bert_USNID_Unsup_9/models",
            "banking": "models/UnsupUSNID/UnsupUSNID_banking_1.0_bert_USNID_Unsup_5/models",
            "snips": "models/UnsupUSNID/UnsupUSNID_snips_1.0_bert_USNID_Unsup_6/models",
            "mcid": "models/UnsupUSNID/UnsupUSNID_mcid_1.0_bert_USNID_Unsup_3/models",
            "dbpedia": "models/UnsupUSNID/UnsupUSNID_dbpedia_1.0_bert_USNID_Unsup_7/models",
        }
        max_seq_lengths = {
            'stackoverflow': 45, 'clinc': 30, 'banking': 55,  
            'snips': 35, 'mcid': 16, 'dbpedia': 14
        }
        dataset_num_labels = {
            'stackoverflow': 20, 'clinc': 150, 'banking': 77,  
            'snips': 7, 'mcid': 16, 'dbpedia': 14
        }

        dataset = dataset.lower()

        if dataset not in usnid_paths:
            raise ValueError(f"Unsupported dataset for UnsupUSNID: '{dataset}'. Available: {list(usnid_paths.keys())}")
        
        model_path = usnid_paths[dataset]
        max_len = max_seq_lengths[dataset]
        num_labels = dataset_num_labels[dataset]
        
        print(f"Loading tokenizer and model from: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        
        mock_args = EasyDict({"num_labels": num_labels, "activation": "tanh"})
        usnid_model = BERT_USNID_UNSUP.from_pretrained(model_path, args=mock_args)
        usnid_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        usnid_model.to(device)

        batch_size = 32
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with UnsupUSNID/{dataset}"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding="max_length",  
                truncation=True, max_length=max_len
            ).to(device)
            
            with torch.no_grad():
                batch_embeddings, _ = usnid_model(**inputs, feature_ext=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        normalized_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        return normalized_embeddings

    elif emb_type.lower() == 'usnid':
        print("Getting embeddings using fine-tuned USNID model...")
        if dataset is None:
            raise ValueError("Parameter 'dataset' must be provided when emb_type is 'usnid'.")
        
        usnid_paths = {
            "stackoverflow": "models/USNID/USNID_stackoverflow_0.75_0.1_1.0_bert_USNID_3/models",
            "clinc": "models/USNID/USNID_clinc_0.75_0.1_1.0_bert_USNID_3/models",
            "banking": "models/USNID/USNID_banking_0.75_0.1_1.0_bert_USNID_3/models",
            "snips": "models/USNID/USNID_snips_0.75_0.1_1.0_bert_USNID_3/models",
            "mcid": "models/USNID/USNID_mcid_0.75_0.1_1.0_bert_USNID_3/models",
            "dbpedia": "models/USNID/USNID_dbpedia_0.75_0.1_1.0_bert_USNID_5/models",
        }

        max_seq_lengths = {
            'stackoverflow': 45, 'clinc': 30, 'banking': 55,  
            'snips': 35, 'mcid': 16, 'dbpedia': 14
        }
        dataset_num_labels = {
            'stackoverflow': 20, 'clinc': 150, 'banking': 77,  
            'snips': 7, 'mcid': 16, 'dbpedia': 14
        }

        dataset = dataset.lower()

        if dataset not in usnid_paths:
            raise ValueError(f"Unsupported dataset for USNID: '{dataset}'. Available: {list(usnid_paths.keys())}")
        
        model_path = usnid_paths[dataset]
        max_len = max_seq_lengths[dataset]
        num_labels = dataset_num_labels[dataset]
        
        print(f"Loading tokenizer and model from: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        
        mock_args = EasyDict({"num_labels": num_labels, "activation": "tanh", "pretrain": False, "wo_self": False})
        usnid_model = BERT_USNID.from_pretrained(model_path, args=mock_args)
        usnid_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        usnid_model.to(device)

        batch_size = 32
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with USNID/{dataset}"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding="max_length",  
                truncation=True, max_length=max_len
            ).to(device)
            
            with torch.no_grad():
                batch_embeddings, _ = usnid_model(**inputs, feature_ext=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        normalized_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        return normalized_embeddings

    else:
        raise ValueError(f"Unsupported emb_type: {emb_type}")

def summarize_cluster(texts, new_cache, old_cache, prompt="", text_type=""):
    if not texts:
        return ""
    if prompt == "":
        prompt = f"Write a single sentence that represents the following cluster concisely:\n\n" + "\n".join(texts)  
    else:
        prompt = prompt + "\n\n" + "\n".join(texts)      
    
    response_text = call_chatgpt(prompt, new_cache, old_cache, num_predictions=1, temperature=0, max_tokens=1000)

    return response_text

def select_representative_samples(embeddings, texts, rep_method='kmeans++', k=5, random_state=None):
    num_samples = len(texts)
    if num_samples == 0:
        return []
    k = min(k, num_samples)

    if rep_method == 'kmeans++':
        _, indices = kmeans_plusplus(np.array(embeddings), n_clusters=k, random_state=random_state)
        return [texts[i] for i in indices]
    elif rep_method == 'mad':
        avg_dist = np.array([np.mean([np.linalg.norm(e1 - e2) for e2 in embeddings]) for e1 in embeddings])
        indices = np.argsort(avg_dist)[-k:]
        return [texts[i] for i in indices]
    elif rep_method == 'mmr':
        selected_indices = []
        if k == 0: return []
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        
        sims_to_centroid = cosine_similarity(embeddings_array, centroid.reshape(1, -1)).flatten()
        first_idx = np.argmax(sims_to_centroid)
        selected_indices.append(first_idx)

        for _ in range(1, k):
            remaining_indices = list(set(range(num_samples)) - set(selected_indices))
            if not remaining_indices: break
            
            sim_to_centroid_rem = sims_to_centroid[remaining_indices]
            
            selected_embs = embeddings_array[selected_indices]
            rem_embs = embeddings_array[remaining_indices]
            
            sim_to_selected = cosine_similarity(rem_embs, selected_embs)
            max_sim_to_selected = np.max(sim_to_selected, axis=1)
            
            mmr_scores = sim_to_centroid_rem - max_sim_to_selected
            best_rem_idx = np.argmax(mmr_scores)
            selected_indices.append(remaining_indices[best_rem_idx])
            
        return [texts[i] for i in selected_indices]
    elif rep_method == 'nn':
        sims = cosine_similarity(embeddings)
        np.fill_diagonal(sims, 0)
        scores = np.sum(sims, axis=1)
        indices = np.argsort(-scores)[:k]
        return [texts[i] for i in indices]
    else:
        raise ValueError(f"Unknown rep_method: {rep_method}")

def enhance_query_with_llm_context(
    text_to_enhance, 
    home_cluster_summary, 
    home_cluster_samples, 
    neighbor_contexts,
    new_cache, 
    old_cache  
):
    home_samples_str = "\n".join([f'- "{s}"' for s in home_cluster_samples])
    home_context_str = (
        f'**Home Cluster (Current Assignment):**\n'
        f'Summary: "{home_cluster_summary}"\n'
        f'Examples:\n{home_samples_str}'
    )

    neighbor_context_str = ""
    for i, neighbor in enumerate(neighbor_contexts):
        neighbor_samples_str = "\n".join([f'- "{s}"' for s in neighbor['samples']])
        neighbor_context_str += (
            f'\n**Neighboring Cluster #{i+1} (Alternative Theme):**\n'
            f'Summary: "{neighbor["summary"]}"\n'
            f'Examples:\n{neighbor_samples_str}\n'
        )

    prompt = (
        "You are a data refinement analyst. Your task is to clarify the true semantic intent of a query based on its surrounding context.\n\n"
        "**Internal Reasoning Process:**\n"
        "1.  **Analyze Context:** Review the 'Home Cluster' (the query's current placement) and the alternative 'Neighboring Clusters'.\n"
        "2.  **Determine Best Fit:** Decide which single cluster theme the 'Query to Refine' aligns with most strongly. This might be the 'Home Cluster' or one of the 'Neighboring Clusters'.\n"
        "3.  **Refine and Distinguish:** Rewrite the query to be a perfect, unambiguous example of the theme you chose in step 2. Ensure the new query is semantically distant from the other cluster themes.\n\n"
        "--- Context ---\n"
        f"{home_context_str}\n"
        f"{neighbor_context_str}"
        "--- End of Context ---\n\n"
        f'**Query to Refine:**\n"{text_to_enhance}"\n\n'
        "**Your Task:**\n"
        "Based on your analysis of the best fit, provide the refined query. Your output must only be the rewritten query text, with no preamble or explanation.\n\n"
        "**Refined Query:**"
    )
    
    return call_chatgpt(prompt, new_cache, old_cache, temperature=0, max_tokens=100)


def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def find_neighbor_proxies(summary_embeddings):
    if summary_embeddings.shape[0] < 2:
        return np.array([], dtype=int)
    sim_matrix = cosine_similarity(summary_embeddings)
    np.fill_diagonal(sim_matrix, -1.0)
    return np.argmax(sim_matrix, axis=1)

def find_k_neighbor_clusters(summary_embeddings, k):
    num_clusters = summary_embeddings.shape[0]
    if num_clusters <= k:
        all_indices = np.arange(num_clusters)
        neighbor_indices = [np.delete(all_indices, i) for i in range(num_clusters)]
        return np.array([arr[:k] if len(arr) >= k else np.pad(arr, (0, k - len(arr)), 'constant', constant_values=-1) for arr in neighbor_indices])
        
    sim_matrix = cosine_similarity(summary_embeddings)
    np.fill_diagonal(sim_matrix, -1.0)
    neighbor_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    return neighbor_indices

def map_summaries_to_known_intents_by_similarity(summaries, summary_embeddings, known_centroids, known_labels):
    print("  [Semi-Supervision] Mapping summaries to known intents by similarity...")
    if summary_embeddings.shape[0] == 0 or known_centroids.shape[0] == 0:
        return {}
        
    similarity_matrix = 1 - cdist(summary_embeddings, known_centroids, 'cosine')
    
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    
    summary_to_known_map = {r: c for r, c in zip(row_ind, col_ind)}
    
    print(f"  [Semi-Supervision] Similarity mapping created for {len(summary_to_known_map)} clusters.")

    for cluster_idx, known_idx in summary_to_known_map.items():
        print(f"    - Mapped Cluster {cluster_idx}: '{summaries[cluster_idx]}' to Known Intent '{known_labels[known_idx]}'")

    return summary_to_known_map

def map_summaries_to_known_intents_with_llm(summaries, known_labels, new_cache, old_cache):
    print("  [Semi-Supervision] Mapping summaries to known intents using LLM (Strict One-to-One)...")
    
    summaries_str = "\n".join([f'Cluster {i}: "{summary}"' for i, summary in enumerate(summaries) if summary])
    known_labels_str = "\n".join([f'- "{label}"' for label in known_labels])

    prompt = (
        "Your task is to create a strict one-to-one mapping from each 'Predefined Intent' to the single most appropriate 'Cluster Summary'.\n\n"
        "**Rules:**\n"
        "1. Every Predefined Intent *must* be mapped to exactly one Cluster Summary.\n"
        "2. A Cluster Summary can only be mapped to one Predefined Intent.\n"
        "3. You must find the best possible pairing for all intents, even if the match is not perfect. Do not leave any intent unmapped.\n\n"
        "**Predefined Intent List:**\n"
        f"{known_labels_str}\n\n"
        "**Cluster Summaries to Map:**\n"
        f"{summaries_str}\n\n"
        "**Instructions:**\n"
        "Provide the final mapping in the format 'Predefined Intent -> Cluster X'. "
        "List every Predefined Intent once. Your output must only contain the mapping lines.\n\n"
        "**Mapping:**\n"
    )

    raw_mapping_str = call_chatgpt(prompt, new_cache, old_cache, temperature=0, max_tokens=2000)


    summary_to_known_map = {}
    used_clusters = set()
    
    if raw_mapping_str:
        print("  [Semi-Supervision] LLM Raw Response:\n" + raw_mapping_str)
        for line in raw_mapping_str.split('\n'):
            if '->' in line:
                try:
                    known_part, cluster_part = line.split('->')
                    known_label = known_part.strip().replace('-', '').replace('"', '').strip()
                    
                    match = re.search(r'Cluster (\d+)', cluster_part)
                    if match and known_label in known_labels:
                        cluster_idx = int(match.group(1))
                        
                        if cluster_idx not in used_clusters:
                            known_label_idx = known_labels.index(known_label)
                            summary_to_known_map[cluster_idx] = known_label_idx
                            used_clusters.add(cluster_idx)
                        else:
                            print(f"  [LLM Parsing Warning] Cluster {cluster_idx} already mapped. Skipping duplicate assignment for '{known_label}'.")
                except (ValueError, IndexError) as e:
                    print(f"  [LLM Parsing Warning] Could not parse line: '{line}'. Error: {e}")

    print(f"  [Semi-Supervision] LLM created {len(summary_to_known_map)} one-to-one mappings.")
    for cluster_idx, known_idx in summary_to_known_map.items():
        print(f"    - Mapped Cluster {cluster_idx}: '{summaries[cluster_idx]}' to Known Intent '{known_labels[known_idx]}'")

    return summary_to_known_map


def DNILC(text_data, 
          num_clusters, 
          init='k-means++', 
          prompt="", 
          text_type="", 
          force_context_length=10, 
          max_llm_iter=5, 
          random_state=None, 
          emb_type='openai', 
          text_features=None,
          true_labels=None,
          instructor_prompt="", 
          rep_method='kmeans++', 
          query_optimization=False, 
          O=10,
          data=None,
          num_neighbor_clusters=2, 
          summary_alpha=1.0,
          summary_beta=0.1,
          summary_gamma=0.1,
          opt_alpha=1.0,
          opt_beta=0.5,
          opt_gamma=0.5,
          use_semi_supervision=False,
          use_initial_seeding=False,
          use_cost_modification=False,
          mapping_strategy='similarity',
          known_centroids=None,
          known_labels=None,
          summary_delta=0.1
         ):
    print(f"Loading new cache from '{NEW_CACHE_FILE}'...")
    new_cache = load_cache(NEW_CACHE_FILE)
    print(f"Loading original cache from '{OLD_CACHE_FILE}'...")
    old_cache = load_cache(OLD_CACHE_FILE)

    query_log = []
    num_samples = len(text_data)
    
    text_data_original = list(text_data) 
    current_text_features = np.copy(text_features)
    
    if true_labels is not None:
        oracle_summaries = {}
    
    print("Initializing clusters with K-Means...")
    kmeans = KMeans(n_clusters=num_clusters, init=init, n_init=1, max_iter=20, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(current_text_features)
    centroids = kmeans.cluster_centers_

    # --- Initial Seeding ---
    if use_semi_supervision and use_initial_seeding and known_centroids is not None and len(known_centroids) > 0:
        print("\n=== Applying Initial Seeding Component ===")
        distance_matrix = cdist(centroids, known_centroids, metric='cosine')
        
        num_known = known_centroids.shape[0]
        if num_clusters >= num_known:
            cluster_indices, known_indices = linear_sum_assignment(distance_matrix)
            
            print(f"  [Seeding] Mapping {len(cluster_indices)} initial clusters to known centroids.")
            for c_idx, k_idx in zip(cluster_indices, known_indices):
                print(f"    - Replacing centroid of Cluster {c_idx} with seed for '{known_labels[k_idx]}'.")
                centroids[c_idx] = known_centroids[k_idx]
        else:
            print("  [Seeding Warning] Number of clusters is less than known intents. Seeding skipped.")
    
    summary_to_known_map = {}

    for iteration in tqdm(range(1, max_llm_iter + 1), desc=f"DNILC Iterations"):
        print(f"\n--- Starting DNILC Iteration {iteration}/{max_llm_iter} ---")
        
        print("Generating cluster summaries with LLM...")
        clustered_texts_for_summary = {i: [] for i in range(num_clusters)}
        for i in range(num_clusters):
            indices_in_cluster = np.where(cluster_assignments == i)[0]
            if len(indices_in_cluster) > 0:
                cluster_embeddings = current_text_features[indices_in_cluster]
                cluster_texts = [text_data_original[j] for j in indices_in_cluster]
                clustered_texts_for_summary[i] = select_representative_samples(
                    cluster_embeddings, cluster_texts, 
                    rep_method=rep_method, 
                    k=min(len(cluster_texts), force_context_length),
                    random_state=random_state
                )
            
        summaries = [summarize_cluster(clustered_texts_for_summary.get(i,[]), new_cache, old_cache, prompt, text_type) for i in range(num_clusters)]
        summary_embeddings = get_embeddings(summaries, emb_type=emb_type, instructor_prompt=instructor_prompt, dataset=data)

        # --- Mapping Strategy ---
        if use_semi_supervision and use_cost_modification and known_centroids is not None:
            print(f"\n=== Applying Mapping Strategy: {mapping_strategy} ===")
            if mapping_strategy == 'llm':
                summary_to_known_map = map_summaries_to_known_intents_with_llm(summaries, known_labels, new_cache, old_cache)
   
            else: # 'similarity'
                summary_to_known_map = map_summaries_to_known_intents_by_similarity(summaries, summary_embeddings, known_centroids, known_labels)
        

        internal_proxies = summary_embeddings
        neighbor_indices_single = find_neighbor_proxies(summary_embeddings)
        external_proxies = np.zeros_like(summary_embeddings)
        if len(neighbor_indices_single) > 0 and neighbor_indices_single.shape[0] == summary_embeddings.shape[0]:
            external_proxies = summary_embeddings[neighbor_indices_single]

        print("Re-assigning points using hybrid geometric-semantic cost...")
        hybrid_costs = np.zeros((num_samples, num_clusters))

        dist_to_all_known_centroids = None
        if use_semi_supervision and use_cost_modification and known_centroids is not None:
             dist_to_all_known_centroids = cdist(current_text_features, known_centroids, metric='cosine')

        for j in range(num_clusters):
            geo_cost = np.linalg.norm(current_text_features - centroids[j], axis=1)**2
            sem_cohesion = 1 - cosine_similarity(current_text_features, internal_proxies[j].reshape(1, -1)).flatten()
            sem_separability_similarity = cosine_similarity(current_text_features, external_proxies[j].reshape(1, -1)).flatten()
            # --- Hybrid Cost Modification ---
            semi_supervision_cost = np.zeros(num_samples)
            if use_semi_supervision and use_cost_modification and j in summary_to_known_map:
                if dist_to_all_known_centroids is not None:
                    known_idx = summary_to_known_map[j]
                    semi_supervision_cost = dist_to_all_known_centroids[:, known_idx]

            hybrid_costs[:, j] = (summary_alpha * geo_cost) + \
                                 (summary_beta * sem_cohesion) + \
                                 (summary_gamma * sem_separability_similarity) + \
                                 (summary_delta * semi_supervision_cost)


        cluster_assignments = np.argmin(hybrid_costs, axis=1)

        print("Updating geometric centroids...")
        for i in range(num_clusters):
            points_in_cluster = current_text_features[cluster_assignments == i]
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)

        # --- Query Optimization Block ---
        if query_optimization:
            print("Identifying and optimizing ambiguous queries...")
            dists_to_centroids = np.linalg.norm(current_text_features[:, None, :] - centroids[None, :, :], axis=2)
            p = softmax(-dists_to_centroids, axis=1)
            entropy = -np.sum(p * np.log(p + 1e-9), axis=1)
            top_idxs = np.argsort(-entropy)[:O]
            
            k_neighbor_indices = find_k_neighbor_clusters(summary_embeddings, k=num_neighbor_clusters)

            print(f"Found {len(top_idxs)} ambiguous queries to refine.")
            for idx in top_idxs:
                original_text = text_data_original[idx]
                old_embedding = np.copy(current_text_features[idx])
                
                home_cluster_idx = cluster_assignments[idx]
                home_cluster_summary = summaries[home_cluster_idx]
                home_cluster_samples = clustered_texts_for_summary.get(home_cluster_idx, [])
                
                neighbor_contexts = []
                if k_neighbor_indices.size > 0 and home_cluster_idx < len(k_neighbor_indices):
                    for neighbor_idx in k_neighbor_indices[home_cluster_idx]:
                        if neighbor_idx != -1: 
                            neighbor_contexts.append({
                                'summary': summaries[neighbor_idx],
                                'samples': clustered_texts_for_summary.get(neighbor_idx, [])
                            })

                enhanced_text = enhance_query_with_llm_context(
                    text_to_enhance=original_text,
                    home_cluster_summary=home_cluster_summary,
                    home_cluster_samples=home_cluster_samples,
                    neighbor_contexts=neighbor_contexts,
                    new_cache=new_cache,
                    old_cache=old_cache
                )

                if not enhanced_text or enhanced_text.strip().lower() == original_text.strip().lower():
                    continue

                new_embedding_array = get_embeddings([enhanced_text], emb_type=emb_type, instructor_prompt=instructor_prompt, dataset=data)
                if new_embedding_array.size == 0:
                    continue
                new_embedding = new_embedding_array[0]

                def calculate_hybrid_cost(embedding_vector):
                    costs = np.zeros(num_clusters)
                    for j in range(num_clusters):
                        geo = np.linalg.norm(embedding_vector - centroids[j])**2
                        coh = 1 - cosine_similarity(embedding_vector.reshape(1, -1), internal_proxies[j].reshape(1, -1)).flatten()[0]
                        sep_sim = cosine_similarity(embedding_vector.reshape(1, -1), external_proxies[j].reshape(1, -1)).flatten()[0]
                        costs[j] = (opt_alpha * geo) + (opt_beta * coh) - (opt_gamma * sep_sim)
                    return costs

                cost_old_vec = calculate_hybrid_cost(old_embedding)
                cost_new_vec = calculate_hybrid_cost(new_embedding)
                
                cost_old = np.min(cost_old_vec)
                cost_new = np.min(cost_new_vec)

                true_label_idx = true_labels[idx] if true_labels is not None else -1
                true_label_info = {}

                print(f"\n> Optimizing query {idx}:")
                print(f"  - Original: '{original_text}'")
                print(f"  - Augmented: '{enhanced_text}'")
                print(f"  - Assigned Cluster: {home_cluster_idx}")
                if true_labels is not None:
                    print(f"  - True Cluster: {true_label_idx}")
                
                print(f"  - Assigned Cluster Summary: '{summaries[home_cluster_idx]}'")

                if true_labels is not None:
                    if true_label_idx not in oracle_summaries:
                        true_indices = np.where(np.array(true_labels) == true_label_idx)[0]
                        if len(true_indices) > 0:
                            true_cluster_texts = [text_data_original[j] for j in true_indices]
                            samples_for_summary = select_representative_samples(current_text_features[true_indices], true_cluster_texts, rep_method=rep_method, k=force_context_length, random_state=random_state)
                            oracle_summaries[true_label_idx] = summarize_cluster(samples_for_summary, new_cache, old_cache, prompt, text_type)
                        else:
                            oracle_summaries[true_label_idx] = " (No samples for this true label)"
                    
                    print(f"  - True Cluster Summary: '{oracle_summaries.get(true_label_idx, 'N/A')}'")
                    true_label_info = {"true_label_idx": true_label_idx}

                if cost_new < cost_old:
                    print("  - Decision: ACCEPTED (new cost is lower)")
                    current_text_features[idx] = new_embedding
                    text_data[idx] = enhanced_text
                    query_log.append({
                        "iteration": iteration, "index": idx, "original": original_text, "optimized": enhanced_text,
                        "decision": "accepted", "cost_old": cost_old, "cost_new": cost_new,
                        "assigned_cluster": home_cluster_idx, **true_label_info
                    })
                else:
                    print("  - Decision: REJECTED (new cost is not lower)")
                    query_log.append({
                        "iteration": iteration, "index": idx, "original": original_text, "optimized": enhanced_text,
                        "decision": "rejected", "cost_old": cost_old, "cost_new": cost_new,
                         "assigned_cluster": home_cluster_idx, **true_label_info
                    })
    
    print(f"Clustering finished. Saving all new and accessed entries to '{NEW_CACHE_FILE}'...")
    save_cache(new_cache, NEW_CACHE_FILE)

    final_centroids = centroids
    final_summaries = summaries
    final_summary_embeddings = summary_embeddings
    
    summaries_evolution = []
    centroids_evolution = []

    return cluster_assignments, final_summaries, final_summary_embeddings, final_centroids, summaries_evolution, centroids_evolution, query_log