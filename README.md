# Sensory Descriptor Identifier

This tool was created by Chreston Miller, a researcher at Virginia Tech, to identify words with flavor-descriptive meanings in English-language food reviews and other texts. It is based on the archetecture described by Intuition Engineering in the article ["Deep learning for specific information extraction from unstructured texts"](https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada). The results on whiskey reviews have been published in ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). Significant changes to streamline and standardize preprocessing have been made by Leah Hamilton, a fellow author and current maintainer of this repository. Email [Chreston Miller](mailto:chmille3@vt.edu) if you want code to identically reproduce the results of the Foods paper.

Programmed in Python using the deep learning backend keras, this package includes a set of whiskey reviews from [Whiskycast](https://whiskycast.com/tastingnotes/), [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/), [The Whiskey Jug](https://thewhiskeyjug.com/), and [Breaking Bourbon](https://www.breakingbourbon.com/bourbon-rye-whiskey-reviews-sort-by-review-date) with manually-annotated examples of flavor-descriptive and flavor-nondescriptive adjectives and nouns, a trained neural network to predict the descriptive or nondescriptive status of a token in context, and code to use the trained network to tag all tokens in the example dataset as descriptive or nondescriptive.

## Installation and Dependencies

When you download this package using `pip`, it should automatically download the dependencies:
- `keras>=2.3`
- `numpy`
- `pandas`
- `tensorflow>2`

If you'd like to enable GPU acceleration (recommended for (re)training), you are encouraged to download your preferred versions of [`tensorflow`](https://www.tensorflow.org/install) (via `pip`) and [CUDA](https://developer.nvidia.com/cuda-toolkit) (via the NVIDIA website) *before* installing this package.

`pip install flavorfetcher`
If you are downloading this package from the source (e.g., acquired via GitHub), you'll have to replace `flavorfetcher` with a pointer to the directory this `README.md` is in, which will be specific to you. As one possible example:
`pip install C:/Users/YOURNAME/Documents/flavor-fetcher`

Once you have installed this package via `pip`, you will be able to run the [code examples below](#predicting-in-python) in any Python interpreter or IDE by [importing the relevant functions](https://docs.python.org/3/reference/import.html).

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
from flavorfetcher.makepredictions import file_to_file_workflow as do_predict
do_predict(glove_file = "glove.42B.300d.txt")
```
Because of its size, the GloVe embedding file is not included with this package and you must *always* specify its location. All other files are optional.

By default, this package will pull the model, settings, and data files from the ones included in the package, and will save the output as `tokenwise_lstm_predictions_YYYYMMDD-HHMMSS.csv` in the current working directory. If you're not sure where that is, you can always specify the output location yourself. You will also likely have to specify your own data, for most use-cases. So a more common usage may look like:
```
from flavorfetcher.makepredictions import file_to_file_workflow as do_predict
do_predict(reviews_file = "example_data.csv", glove_file = "glove.42B.300d.txt", output_file = "predicted_descriptors.csv")
```
In practice, you will likely need to change the above file names to reflect your specific use-case and to represent where the files are located in your filesystem, unless they're all in the current working directory.

The script, by default, looks for [the exact column names mentioned above](#data-formatting). All files need to have these columns, but if your `.csv` file has differing column *names*, you can specify them as follows:
```
mycolnames = {"token" : "customtokenid", "upos" : "customuposid", "doc_id" : "customdocid", "sid" : "customsentenceid", "tid" : "customtokenid"}
file_to_file_workflow("example_data.csv", "glove.42B.300d.txt", "model.h5", "model.json", "predicted_descriptors.csv", mycolnames)
```

You can also change only a few of the default column names by importing `flavorfetcher.datamunging.default_col_names`:
```
from flavorfetcher.datamunging import default_col_names as mycolnames
mycolnames['tid'] = "token_id"
mycolnames['sid'] = "sentence_id"
file_to_file_workflow("example_data.csv", "glove.42B.300d.txt", "model.h5", "model.json", "predicted_descriptors.csv", mycolnames)
```

### Predicting in the Command Line

*Coming soon*

### Predicting in R

R and RStudio users should be able to use the R package [`reticulate`](https://rstudio.github.io/reticulate/) to call this package's functions from within an R script, but this functionality has not been tested and should be considered a recommendation for advanced users only.

*A full tutorial for R users is planned in the future.*

### Training

The `descriptor_model` module contains a definition of the `DescriptorNN` class, whose methods will allow for limited training, but anyone looking to change the architecture or fine-tune the model should look into implementing a model of the same/similar structure in `keras` itself.

## Contributing

If you use this tool, the best ways to give back are to [cite our paper](https://doi.org/10.3390/foods10071633) and to let us know how it's doing! We are actively looking to improve this tool.

To let us know how it's doing, take around 20 sentences from your dataset (or more), tokenize them, and put them into a [tidy format](#data-formatting). Add a column with the "ground truth"--`0` for most of the tokens and `1` if you think it is a word describing a flavor. Run this little dataset through our tool and send the results to [Leah Hamilton](mailto:lmhamilton@ucdavis.edu) with a little explanation of your data/project. We may contact you about your data if the tool is performing poorly on it, to help make it better for everyone. Even if this isn't possible, we appreciate you sending a little report one way or another!

This repository only works on English text. If you are interested in identifying descriptors in another language *and* speak that language, please reach out to [Leah Hamilton](mailto:lmhamilton@ucdavis.edu) about a potential collaboration.

## Troubleshooting

### Python Setup

This package requires python 3.7 or newer. If you don't know how python works, how to install it, or how to run python code, you should start by reading and following the installation tutorial for [Windows](https://docs.python.org/3/using/windows.html) or [Mac](https://docs.python.org/3/using/mac.html) computers *and* the [tutorial on using python interactively](https://docs.python.org/3/tutorial/interpreter.html). We advise *against* installing python as a part of `anaconda`, since it will download much more than you actually need.

If you are having issues [installing this package](#installation-and-dependencies) or if you get any error messages related to `keras`, `tensorflow`, `pip`, or `setuptools`, especially if you're using a Mac, you may find that it's easier to install and run this package inside of a virtual environment. Note that some `tensorflow` "errors" are more like warnings--if the code continues to run, it's probably fine!

A [python virutal environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) is a quarantined installation of python that allows you to have more control over exactly what versions of what packages are installed without interfering with or trying to use other python installations on your system.

First, decide where you'll save your virtual environment. This could be in your documents folder, or wherever you keep your code. Whatever is convenient and memorable for you. Navigate there in your command line of choice.

To create the virtual environment here and install the packages needed, run the code:
```
python -m venv flavor-fetcher-test
flavor-fetcher-test\Scripts\activate
python -m pip install flavorfetcher
```
This only needs to be run once per system, when you first set up your virtual environment. You can change the name of the virtual environment (`flavor-fetcher-test`) if you want. You will have to make the corresponding change in the line that activates the virtual environment as well.

From here, once the virtual environment is activated (you'll know because the command line will replace the `>` character with the name of the virtual environment wherever you're typing), you can start a python session using `python` and then follow [the instructions above](#predicting-in-python) to actually run the package.

Any time you want to use the virtual environment, you should run `flavor-fetcher-test\Scripts\activate`, and when you are done, you can deactivate it by typing `deactivate`.

### Speed

On a middle-of-the-road computer, you should estimate that running this package will take about 5 minutes for initial setup and then 20-50 ms/token to make the predictions.

If you have a particularly large dataset, you might consider splitting it into multiple files, as the program does not save until it's finished and you will lose all prediction progress on an interruption.

If you have a computer with an NVIDIA GPU, you can also set up `tensorflow` to use GPU acceleration. To do this, you should install an appropriate version of CUDA and a GPU-compatible version of `tensorflow` yourself using `pip` as descibed [in the tensorflow documentation](https://www.tensorflow.org/install/pip), rather than letting this package install it as a dependency. If you are using a laptop and are not sure whether you have an NVIDIA GPU, you probably don't.

## Files
- `README.md` - This file.
- `LICENSE` - The GPL v3.0 license this software is provided under.
- `MANIFEST.in` & `pyproject.toml` - Files required for `pip` or another builder to make an installable version of this code. Do not move or change these.
- `src/makepredictions.py` - A python module with the function `file_to_file_workflow()`, which can be given the file locations of a `.csv` reviews_file, a `.h5` model file with the appropriate architecture, the 300-dimension [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (`glove.42B.300d.txt`), a settings file (`.json` with the `CONTEXT_WINDOW_SIZE` and `GLOVE_EMBEDDING_DIMENSIONS` used to train the model), and an output file (`.csv`, should not currently exist). If the column names are not identical to those in the example file `src/ata/Example Whiskey Reviews.csv`, the column names will have to be specified as a dictionary using the last argument as show in the usage examples. This file can also be run as a script from the command line, which will by default reproduce the results of ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633) but can be used to use a pre-trained model to predict descriptors in any tokenized and POS-tagged dataset with the appropriate arguments.
- `src/datamunging.py` - A python module with many helper functions used to remove punctuation, reindex tokens for `keras`, convert tokens to GloVe embeddings, and set each token's context window. Primarily called by `src/makepredictions.py`
- `src/descriptor_model.py` - A python script defining the `DescriptorNN` class, which holds a (trained) descriptor-predictor model and is used to make predictions.
- `src/data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5` - The exact network trained and used in the paper ["Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning"](https://doi.org/10.3390/foods10071633). Necessary to make predictions using `src/makepredictions.py`.
- `src/data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json` - A settings file that should be saved and distributed alongside the trained model in order to specify the `CONTEXT_WINDOW_SIZE` and `GLOVE_EMBEDDING_DIMENSIONS` used to train the model so predictions can be made on appropriately-shaped data.
- `src/data/Example Whiskey Reviews.csv` - A data file containing one row per token in 100 selected reviews from the [Whiskycast](https://whiskycast.com/tastingnotes/) and [WhiskyAdvocate](https://www.whiskyadvocate.com/ratings-and-reviews/) websites, used for prediction. Tokenization and annotation done using SpaCy through CleanNLP. Only the fields `upos`, `token`, `doc_id`, `sid`, and `tid` are used. It also contains the predictions as calculated by the provided LSTM on our machine (`expected_prediction`), so you can confirm the package is working as intended.

## Citing this Package
If you use this package in any published work, please cite the associated publication:

Miller, C.; Hamilton, L.; Lahne, J. 2021. Sensory Descriptor Analysis of Whisky Lexicons through the Use of Deep Learning. *Foods* 10(7), 1633. https://doi.org/10.3390/foods10071633 