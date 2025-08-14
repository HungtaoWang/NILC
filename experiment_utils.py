from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def avg_closest_distance(emba, embb):
    distance_matrix = cdist(emba, embb, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    assigned_distances = distance_matrix[row_ind, col_ind]
    average_closest_distance = np.mean(assigned_distances)
    return average_closest_distance

def load_dataset(type, opt=None):
    if type == 'banking':
        return load_banking()
    elif type == 'clinc':
        return load_clinc()
    elif type == 'stackoverflow':
        return load_stackoverflow()
    elif type == 'snips':
        return load_snips()
    elif type == 'dbpedia':
        return load_dbpedia()
    elif type == 'mcid':
        return load_mcid()

def load_banking():
    instructor_prompt = 'Represent this online banking question for clustering:'
    prompt = "The following is a cluster of online banking questions. Write a single question that represents the cluster concisely."
    text_type = 'Intent:'
    num_clusters = 77
    data = pd.read_csv("banking.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def load_clinc():
    instructor_prompt = 'Represent this question for clustering:'
    prompt = "The following is a cluster of user queries. Write a single question that represents the cluster concisely."
    text_type = 'Intent:'
    num_clusters = 150
    data = pd.read_csv("clinc.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def load_stackoverflow():
    instructor_prompt = 'Represent this technical question for clustering:'
    prompt = "The following is a cluster of technical questions. Write a single question that represents the cluster concisely."
    text_type = 'Intent:'
    num_clusters = 20
    data = pd.read_csv("stackoverflow.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def load_snips():
    instructor_prompt = 'Represent this voice command for clustering:'
    prompt = "The following is a cluster of voice commands. Write a single command that represents the cluster concisely."
    text_type = 'Intent:'
    num_clusters = 7
    data = pd.read_csv("snips.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def load_dbpedia():
    instructor_prompt = 'Represent this text snippet for clustering based on its entity type:'
    prompt = "The following is a cluster of text snippets describing a single type of entity. Write a single, concise label for the entity type."
    text_type = 'Intent:'
    num_clusters = 14
    data = pd.read_csv("dbpedia.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def load_mcid():
    instructor_prompt = 'Represent this COVID-19 related user query for clustering:'
    prompt = "The following is a cluster of COVID-19 related user queries. Write a single question that represents the cluster concisely."
    text_type = 'Intent:'
    num_clusters = 16
    data = pd.read_csv("mcid.tsv", sep="\t")
    return list(data['label']), list(data['text']), num_clusters, prompt, text_type, instructor_prompt

def cluster_metrics(y_true, y_pred, centroid_true, centroid_pred, summary_true, summary_pred=None):
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    from scipy.optimize import linear_sum_assignment as hungarian
    row_ind, col_ind = hungarian(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
    
    cen_cen_dist = avg_closest_distance(centroid_pred, centroid_true)
    cen_sum_dist = avg_closest_distance(centroid_pred, summary_true)
    
    if summary_pred is None:
        summary_pred = centroid_pred
        
    sum_cen_dist = avg_closest_distance(summary_pred, centroid_true)
    sum_sum_dist = avg_closest_distance(summary_pred, summary_true)
    
    return [acc, nmi, ari, cen_cen_dist, cen_sum_dist, sum_cen_dist, sum_sum_dist]