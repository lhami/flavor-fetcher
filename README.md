# Sensory Descriptor Identifier

This tool was created by Chreston Miller, a researcher at Virginia Tech, to identify words with flavor-descriptive meanings in food reviews and other texts. It is based on the archetecture described by Intuition Engineering in the article ["Deep learning for specific information extraction from unstructured texts"](https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada). The results on whiskey reviews have been published in ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). Significant changes to streamline and standardize preprocessing have been made by Leah Hamilton, a fellow author and current maintainer of this repository. Email [Chreston Miller](mailto:chmille3@vt.edu) if you want code to identically reproduce the results of the Foods paper.

Programmed in Python using the deep learning backend keras, this package includes a set of whiskey reviews from [Whiskycast](https://whiskycast.com/tastingnotes/), [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/), [The Whiskey Jug](https://thewhiskeyjug.com/), and [Breaking Bourbon](https://www.breakingbourbon.com/bourbon-rye-whiskey-reviews-sort-by-review-date) with manually-annotated examples of flavor-descriptive and flavor-nondescriptive adjectives and nouns, a trained neural network to predict the descriptive or nondescriptive status of a token in context, and code to use the trained network to tag all tokens in the example dataset as descriptive or nondescriptive.

## Installation and Dependencies

When you download this package using `pip`, it should automatically download the dependencies:
- keras>=2.3
- numpy
- pandas

`pip install flavor-fetcher`

Additionally, to run this code, you will need to manually download the 300-dimension [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) from the Stanford NLP website (`glove.42B.300d.txt`). If you're new to programming, we recommend saving this large file to the same place as your data files so you'll know where to find it.

## Usage

This code is intended to replicate the results from ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633) or allow users to identify flavor-descriptors in their own datasets using the pre-trained model.

### Data Formatting

This neural network was trained using text that had been tokenized by the R package cleanNLP, a "tidy" implementation of SpaCy, and specifically using the language model `en_core_web_sm`.
If you have your data in the format created by cleanNLP's implementation of SpaCy, it should already be formatted appropriately, i.e., with columns:
- `token` - The token as a string, lowercase and with white space removed but otherwise as it appears in the text.
- `upos` - The part of speech, based on the Universal Part of Speech bank
- `doc_id` - A unique number corresponding to the document that the token was in.
- `sid` - A number indicating which sentence the token was in within its document.
- `tid` - A number indicating a token's position within its sentence.

You can use any English-language tokenizer & part-of-speech tagger to prepare your data (although results may vary based on tokenization differences) so long as you then format your data into a `.csv` file with these columns. If your data has these columns but not these exact names, you can provide this tool with a special column specification as shown below.

### Predicting in Python

You can make predictions in Python by importing the package and calling the function `file_to_file_workflow()` from the module `flavor-fetcher.makepredictions`. The simplest example (using the provided example data, model, etc) is:
```
from flavor-fetcher.makepredictions import file_to_file_workflow
file_to_file_workflow("example_data.csv", "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5", "glove.42B.300d.txt", "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json", "predicted_descriptors.csv")
```

In practice, you will likely need to change the file names to reflect your specific use-case and to represent where the files are located in your filesystem, unless they're all in the current working directory.
The script, by default, looks for [the exact column names mentioned above](###Data Formatting). All files need to have these columns, but if your `.csv` file has differing column names, you can specify them as follows:

```
mycolnames = {"token" : "customtokenid", "upos" : "customuposid", "doc_id" : "customdocid", "sid" : "customsentenceid", "tid" : "customtokenid"}
file_to_file_workflow("example_data.csv", "model.h5", "glove.42B.300d.txt", "model.json", "predicted_descriptors.csv", mycolnames)
```

You can also change only a few of the default column names by importing `flavor-fetcher.datamunging.default_col_names`:

```
from flavor-fetcher.datamunging import default_col_names as mycolnames
mycolnames['tid'] = "token_id"
mycolnames['sid'] = "sentence_id"
file_to_file_workflow("example_data.csv", "model.h5", "glove.42B.300d.txt", "model.json", "predicted_descriptors.csv", mycolnames)
```

### Predicting in the Command Line



### Training

The `descriptor_model` module contains a definition of the `DescriptorNN` class, whose methods will allow for limited training, but anyone looking to change the architecture or fine-tune the model should look into implementing a model of the same/similar structure in `keras` itself.

## Files
- `README.md` - This file.
- `LICENSE` - The GPL v3.0 license this software is provided under.
- `src/makepredictions.py` - A python module with the function `file_to_file_workflow()`, which can be given the file locations of a `.csv` reviews_file, a `.h5` model file with the appropriate architecture, the 300-dimension [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (`glove.42B.300d.txt`), a settings file (`.json` with the `CONTEXT_WINDOW_SIZE` and `GLOVE_EMBEDDING_DIMENSIONS` used to train the model), and an output file (`.csv`, should not currently exist). If the column names are not identical to those in the example file `data/annotated_whisky_reviews_032320.csv`, the column names will have to be specified as a dictionary using the last argument as show in the usage examples. This file can also be run as a script from the command line, which will by default reproduce the results of ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633) but can be used to use a pre-trained model to predict descriptors in any tokenized and POS-tagged dataset with the appropriate arguments.
- `src/datamunging.py` - A python module with many helper functions used to remove punctuation, reindex tokens for `keras`, convert tokens to GloVe embeddings, and set each token's context window. Primarily called by `src/makepredictions.py`
- `src/descriptor_model.py` - A python script defining the `DescriptorNN` class, which holds a (trained) descriptor-predictor model and is used to make predictions.
- `data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5` - The exact network trained and used in the paper ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). Necessary to make predictions using `src/makepredictions.py`.
- `data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json` - A settings file that should be saved and distributed alongside the trained model in order to specify the `CONTEXT_WINDOW_SIZE` and `GLOVE_EMBEDDING_DIMENSIONS` used to train the model so predictions can be made on appropriately-shaped data.
- `data/annotated_whisky_reviews_032320.csv` - A data file containing one row per token in the [Whiskycast](https://whiskycast.com/tastingnotes/) and [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/) reviews, used for prediction. Tokenization and annotation done using SpaCy through CleanNLP. Only the fields `upos`, `token`, `doc_id`, `sid`, and `tid` are used.