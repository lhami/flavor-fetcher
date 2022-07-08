# Sensory Descriptor Identifier

This tool was created by Chreston Miller, a researcher at Virginia Tech, to identify words with flavor-descriptive meanings in food reviews and other texts. It is based on the archetecture described by Intuition Engineering in the article ["Deep learning for specific information extraction from unstructured texts"](https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada). The results on whiskey reviews have been published in ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633).

Programmed in Python using the deep learning backend keras, this package includes a set of whiskey reviews from [Whiskycast](https://whiskycast.com/tastingnotes/), [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/), [The Whiskey Jug](https://thewhiskeyjug.com/), and [Breaking Bourbon](https://www.breakingbourbon.com/bourbon-rye-whiskey-reviews-sort-by-review-date) with manually-annotated examples of flavor-descriptive and flavor-nondescriptive adjectives and nouns, code to train a neural network to predict the descriptive or nondescriptive status of a token in context, the trained network itself, and code to use the trained network to tag all tokens in the example dataset as descriptive or nondescriptive.

## Usage

This code is intended to replicate the results from ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). If you wish to update the code to train a new predictor or make and test predictions on a new set of data, you will have to make modifications to the code itself. We plan to convert this repository to R code in the future.

## Files
- `README.md` - This file.
- `100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5` - The exact network trained and used in the paper ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). Necessary to make predictions using `CollectPredictions.py` or `CollectPredictions.ipynb`.
- `annotated_whisky_reviews_032320.csv` - A data file containing one row per token in the [Whiskycast](https://whiskycast.com/tastingnotes/) and [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/) reviews, used for prediction. Tokenization and annotation done using SpaCy through CleanNLP. Only the fields `upos`, `token`, `doc_id`, `sid`, and `tid` are used.
- `CollectPredictions.py` - A python script that, given a pretrained model file (`*.h5`), a copy of GloVe embeddings (`glove.42B.300d.zip`), and a set of tokenized review data as a one-line-per-token `.csv` (with columns `upos`, `token`, `doc_id`, `sid`, and `tid`, as output from e.g. SpaCy), will output a prediction for each individual token in context (0-1, with 0 being nondescriptive and 1 being descriptive).
- `CollectPredictions.ipynb` - A jupyter notebook version of the code in `CollectPredictions.py`.
- `descriptor_model.py` - A python script defining the `DescriptorNN` class, which holds a (trained) descriptor-predictor model and is used to make predictions.
- `train.py` - A python script to train a `DescriptorNN` based on a `.csv` containing examples of annotated tokens in context (Note: the version currently here is out of date, I need to email Dr. Miller)
- `library.py` - Various helper functions used in `CollectPredictions.*` and `descriptor_model.py`
- `wawc_whiskey_reviews_tokenwise_lstm_predictions.csv` - A data file containing the output of `CollectPredictions.*` when used to make predictions on `annotated_whisky_reviews_032320.csv`
