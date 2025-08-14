# config for USNID on CLINC
config = {
    "use_semi_supervision": [True],
    "use_initial_seeding": [True],
    "use_cost_modification": [True],
    "mapping_strategy": ['similarity'],
    "max_llm_iter": [3],
    "rep_method": ['mmr'],
    "query_optimization": [True],
    "summary_beta": [0.5],
    "summary_gamma": [0.5],
    "summary_delta": [0.5],
    "O": [10],
    "num_neighbor_clusters": [10],
    "seeds": [[0]], 
}