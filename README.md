# NILC

This repository contains the source code, data, and models in the paper. 

## Setup Instructions

To set up the environment and install the required dependencies, please follow these steps：

```
conda create -n NILC -c conda-forge python=3.12
conda activate NILC
pip install -r requirements.txt
```

## Models

Before processing the data or running experiments, you can choose to either download our pre-trained models or configure your own.

### Our Pre-trained Models

Our fine-tuned `USNID` and `UnsupUSNID` models can be downloaded from the link below:

- **Fine-tuned Models:** https://mega.nz/folder/Sew0kLzR#CdX4euhBMuGC_xiA3T1Hfg

Put the `models` in the root directory.

### Using Other Encoders

Our framework is designed to be flexible and extensible. You can easily integrate other models as encoders with minimal code modifications. We have built-in support for some strong **previous works** (**USNID**, **MTP-CLNN**, **LatentEM**) and several popular **embeddings** (**SentenceBERT**, **Instructor**, **OpenAI**), including but not limited to:

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

- **Pre-processed Data:** https://mega.nz/folder/PbZ3ED6R#MdBPcWcqTkgTQuBZyEi-KQ

Put the `processed_data` in the root directory.

### Option 2: Pre-process Data Manually

If you are using a different model or want to generate the data embeddings yourself, follow these steps:

```
cd data_loaders
python preprocess_offline_data.py
```

## Running the Experiments

Follow these steps to run the main experiments.

### Step 1: Configure the Experiment

Open `config.py` in the root directory. Set the `EMBEDDING_TYPE` and `DATASET_NAME` variables.

### Step 2: Run the Experiment Script

Execute the main experiment script from the root directory.

```
python run_experiments.py
```

### Step 3: Process the Results

Once all experiments are complete, navigate to the `results` directory and run the processing script to generate a consolidated summary of all results.

```
cd results
python process_results.py
```
