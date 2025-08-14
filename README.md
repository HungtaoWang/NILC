# NILC

This repository contains the source code, data, and models in the paper. 

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Models](#models)
- [Data](#data)
- [Running the Experiments](#running-the-experiments)

## Setup Instructions

To set up the environment and install the required dependencies, please follow these steps.

1. **Create a Conda Environment:**

   ```
   conda create -n NILC -c conda-forge python=3.12
   ```

2. **Activate the Environment:**

   ```
   conda activate NILC
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

## Models

Before processing the data or running experiments, you can choose to either download our pre-trained models or configure your own.

### Our Pre-trained Models

Our fine-tuned `USNID` and `UnsupUSNID` models can be downloaded from the link below:

- **Fine-tuned Models:** [TODO: Add download link here]

### Using Other Encoders

Our framework is designed to be flexible and extensible. You can easily integrate other models as encoders with minimal code modifications. We have built-in support for several popular models, including but not limited to:

- **USNID / UnsupUSNID**: Pre-train or fine-tune your own versions from the [official repository](https://github.com/thuiar/TEXTOIR).
- **MTP-CLNN**: From the [official repository](https://github.com/fanolabs/NID_ACLARR2022).
- **LatentEM**: From the [official repository](https://github.com/zyh190507/Probabilistic-discovery-new-intents).
- **SentenceBERT**
- **Instructor**
- **OpenAI Embeddings**

## Data

With a model selected, you can now prepare the data.

### Option 1: Download Pre-processed Data

You can download the data already pre-processed by our fine-tuned `USNID` and `UnsupUSNID` models from the following link.

- **Pre-processed Data:** [TODO: Add download link here]

### Option 2: Pre-process Data Manually

If you are using a different model or want to generate the data embeddings yourself, follow these steps. The raw data is included in the `data_loaders` directory.

1. Navigate to the data loading directory:

   ```
   cd data_loaders
   ```

2. Run the preprocessing script. This script will use the selected model to generate the necessary data files.

   ```
   python preprocess_offline_data.py
   ```

## Running the Experiments

Follow these steps to replicate the main experiments.

### Step 1: Configure the Experiment

Open `config.py` in the root directory. Set the `EMBEDDING_TYPE` and `DATASET_NAME` variables.

### Step 2: Run the Experiment Script

Execute the main experiment script from the root directory:

```
python run_experiments.py
```

### Step 3: Process the Results

Once all experiments are complete, navigate to the `results` directory and run the processing script to generate a consolidated summary of all results.

```
cd results
python process_results.py
```
