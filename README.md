#IU NLP Medical Text Classification#

This repository collects resources for supervised topic classification of medical abstracts, combining a pure-Python Multinomial Naive Bayes baseline with a BioMedBERT fine-tuning notebook.

##Repository layout##
Multinomial_Naive_Bayes/topic_modeling/ – dataset loaders, preprocessing utilities, a custom Multinomial Naive Bayes classifier, and experiment orchestration helpers for classical topic classification workflows.

Multinomial_Naive_Bayes/run_experiments.py – command-line entry point that wires the components together and drives end-to-end experiments.

FineTunedBioMedBERT/FineTuneBioMedBert.ipynb – notebook for transformer-based fine-tuning; TensorBoard logs from training runs are stored alongside it.

data/ – curated CSV splits of medical abstracts with label 5 removed, plus the original raw exports used as a fallback when the curated path is unavailable.

##Getting started##
Create the recommended Conda environment and install dependencies:

conda env create -f environment.yml
conda activate iu-nlp-topic-modeling
The environment specification pins Python 3.10.10 and the core libraries required for both the classical and transformer pipelines.

###Data###
Experiments expect the curated medical abstracts dataset under data/data_excluding_5, which removes the deprecated condition label 5 and provides aligned train/test CSVs along with a label-name mapping. If that directory is missing, the loader automatically falls back to data/raw so long as the original exports are present.

##Running the Multinomial Naive Bayes baseline##
Run the classical pipeline from the repository root:

python Multinomial_Naive_Bayes/run_experiments.py

The CLI accepts optional --training-sizes, --output-dir, and --random-state arguments, clipping oversize subsets and interpreting full as the complete training set. Each invocation iterates over the requested subset sizes, trains the Naive Bayes model, evaluates macro-averaged metrics, and appends results to experiment_metrics csv while also saving a final classification_report.csv in the chosen output directory.

##Experiment flow##
Preprocessing – Text is lowercased, stop words are removed, lightweight lemmatization rules are applied, and a sorted vocabulary is built before converting abstracts to sparse bag-of-words vectors.

Sampling – iter_size_progression deduplicates and caps requested training sizes, while sample_documents draws deterministic subsets without replacement to study data-efficiency.

Modeling – A custom Multinomial Naive Bayes implementation estimates smoothed log priors and feature likelihoods, then scores test documents in log space for stability.

Metrics – Accuracy plus macro precision, recall, and F1 are computed manually to avoid extra dependencies, and per-class precision/recall/F1/support are exported for the largest training run.

##BioMedBERT fine-tuning##
Open FineTunedBioMedBERT/FineTuneBioMedBert.ipynb to run transformer-based experiments; the directory’s TensorBoard event files capture previous training runs and can be inspected for learning curves.

