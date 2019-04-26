# Instructions
## 1. Preprocessing
* Feedback and Concepts extraction from files into dataframes.
* Left Join Feedback and Concepts on SNO and pull out all unmatched feedback text used to score.
* Inner Join Feedback and Concepts on SNO and pill out all matched feedback text using for training
* Convert feedback text to vector using Gensim Doc2vec and cluster the unmatched feedback text under each label/issue, each   label/issue will have 0 to K clusters. A concept is generated from cluster.
* Gensim Extractive summarization on the cluster to make the feedback text to 400 words.
* Aguments private data with publicly available cnn-dailymail stories dataset.
* Create .story file for each cluster of each label so that these files can be fed to cnn-dailymail story file processor.


## 2. Download cnn-dailymail stories data
Download and unzip the stories directories from https://cs.nyu.edu/~kcho/DMQA/ for both CNN and Daily Mail.

Warning: These files contain a few (114, in a dataset of over 300,000) examples for which the article text is missing - see for example cnn/stories/72aba2f58178f2d19d3fae89d5f3e9a4686bc4bb.story. The Tensorflow code has been updated to discard these examples.

## 3. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it from https://stanfordnlp.github.io/CoreNLP/ and unzip it. Then add the following command to your bash_profile:

export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

replacing /path/to/ with the path to where you saved the stanford-corenlp-full-2016-10-31 directory. You can check if it's working by running

echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

You should see something like:
Please
toeknize
this
text
.
PTBTokenizer tokenized 5 tokens at 68.97 tokens per second.

## 4. Process into .bin and vocab files
Run

python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories
replacing /path/to/cnn/stories with the path to where you saved the cnn/stories directory that you downloaded; similarly for dailymail/stories.

This script will do several things:

* The directories cnn_stories_tokenized and dm_stories_tokenized will be created and filled with tokenized versions of cnn/stories and dailymail/stories. This may take some time. Note: you may see several Untokenizable: warnings from Stanford Tokenizer. These seem to be related to Unicode characters in the data; so far it seems OK to ignore them.

* For each of the url lists all_train.txt, all_val.txt and all_test.txt, the corresponding tokenized stories are read from file, lowercased and written to serialized binary files train.bin, val.bin and test.bin. These will be placed in the newly-created finished_files directory. This may take some time.

* Additionally, a vocab file is created from the training data. This is also placed in finished_files.

* Lastly, train.bin, val.bin and test.bin will be split into chunks of 1000 examples per chunk. These chunked files will be saved in finished_files/chunked as e.g. train_000.bin, train_001.bin, ..., train_287.bin. This should take a few seconds. You can use either the single files or the chunked files as input to the Tensorflow code (see considerations here).

## 5. Concept Generation using deep learning pointer-generator networks
* Once the .bin files are generated in which we have private and public data, we train the pointer-generator with coverage model on training bin files.
* We can start validation process as well along with training process in order to keep track of best model and to terminate the training at the right time.
* using the trained pointer-generator with coverage model we can run the decode/scoring process for the feedback cluster that does not have concept and deep learning network will generate a concept per feedback cluster per label.
