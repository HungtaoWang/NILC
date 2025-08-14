# config for UnsupUSNID on STACKOVERFLOW
config = {
    "use_semi_supervision": [False],
    "max_llm_iter": [3],
    "rep_method": ['mmr'],
    "query_optimization": [True],
    "summary_beta": [0.9],
    "summary_gamma": [0.1],
    "summary_delta": [0.5], # Placeholder
    "O": [10],
    "num_neighbor_clusters": [10],
    "seeds": [[7]],
}