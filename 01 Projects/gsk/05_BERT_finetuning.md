# Fine-Tuning BERT for Feelings Classification

> **Notebook:** [Bert_Multiclass_Feelings_Training.py](../Databricks/notebooks/-deprecated-/Feelings_Multiclass/Bert_Multiclass_Feelings_Training.py)  
> **Status:** Deprecated training workflow; this document describes the code as it exists in the repository, not a currently supported production process.

## Objective

Fine-tune a pretrained, uncased BERT encoder to classify medical-insight text into one of nine
emotion/communication-tone classes: **Anticipation, Concerned, Descriptive, Dissatisfied,
Frustrated, Impressed, Inquisitive, Satisfied,** or **Skeptical**. The notebook treats this as
single-label, multi-class text classification: each input text receives one class label. This is
distinct from the active three-way positive/negative/neutral sentiment classifier.

The intended model can support downstream interpretation of how an insight is expressed. The
notebook itself trains, evaluates on a validation split, and predicts on a held-out test split; it
does not define an online serving endpoint or document a production deployment step.

## Theory: Why Fine-Tune BERT?

**BERT** (Bidirectional Encoder Representations from Transformers) is a Transformer encoder
pretrained on large text corpora. Its self-attention layers build contextual token representations
using both left and right context. As a result, a token representation can reflect the surrounding
phrase rather than only a fixed dictionary meaning.

Fine-tuning adapts those pretrained weights to a labeled task. Here, the notebook uses the pooled
representation of the input sequence as a whole-text representation, then learns a small
classification head over the nine emotion labels. During training, those head weights and the BERT
encoder are trainable (`trainable=True`). This transfer-learning approach typically needs less
labeled task data than training a language model from scratch, while still allowing the
representation to specialize to the domain and label scheme.

The classifier produces one score per class. Softmax/log-softmax converts those scores into a
normalized class distribution; the predicted class is the one with the highest score. Training
minimizes categorical cross-entropy, which penalizes probability assigned away from the known label.
Dropout is applied to the pooled representation during training to reduce overfitting.

## Methodology

### 1. Data source and access

- Mounts the `rdmiblob` Azure Data Lake filesystem using OAuth credentials retrieved from the
	Databricks secret scope `databricks-akv`.
- Reads the labeled CSV at `Saumya/Feelings/data_feelings.csv` beneath that mount.
- Expects at least `text` and `emotion` columns. The labels are learned from the values in
	`emotion` using scikit-learn's `LabelEncoder`.
- Writes model checkpoints and TensorFlow Estimator output beneath
	`/dbfs/mnt/rdmiblob/Saumya/Feelings/outputs_tf`.

### 2. Text cleaning and dataset split

`clean_text()` removes HTML markup with BeautifulSoup, removes @-mentions and URLs with regular
expressions, retains ASCII letters plus `.`, `!`, `?`, and apostrophes, then collapses repeated
spaces. The notebook applies `LabelEncoder` to `emotion` and creates a numeric `labels` column.

The code first holds out 20% of the rows, then splits that held-out portion 80/20 into validation
and test. This results in **80% train / 16% validation / 4% test**. Both calls use
`random_state=100`; neither specifies `stratify`, so preservation of each class's proportions is
not guaranteed. The notebook plots training-label counts but does not implement class weighting or
resampling.

### 3. BERT input preparation

- Loads the pretrained TensorFlow Hub model `bert_uncased_L-12_H-768_A-12/1` (12 encoder layers,
	hidden size 768, 12 attention heads; uncased vocabulary).
- Uses the model's vocabulary and lowercasing setting to initialize Google's BERT
	`FullTokenizer`.
- Wraps each example as `text_a` plus its integer label; `text_b` is unused because this is
	single-sequence classification.
- Converts examples into BERT's `input_ids`, `input_mask`, and `segment_ids` features, with
	`MAX_SEQ_LENGTH = 40`. Inputs longer than the configured sequence limit are constrained by the
	feature conversion; shorter inputs are padded and the mask distinguishes padding from real
	tokens.

### 4. Classification head and objective

The notebook calls the TensorFlow Hub BERT module with `trainable=True` and uses its `pooled_output`
for sequence classification. It adds a learned linear layer with nine output logits and bias,
applies dropout with `keep_prob=0.9`, and computes log-softmax probabilities. Integer labels are
converted to one-hot vectors; the loss is mean categorical cross-entropy:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{9} y_{ik}\log p_{ik}
$$

where $y_{ik}$ is 1 when example $i$ has class $k$, and $p_{ik}$ is the model's predicted
probability for that class. The optimizer is created by `bert.optimization.create_optimizer`,
which applies the configured learning rate and warmup schedule.

### 5. Training and validation

| Setting | Value |
|---|---:|
| Batch size | 8 |
| Learning rate | `2e-5` |
| Training epochs | 5 |
| Warmup proportion | 0.1 of training steps |
| Maximum sequence length | 40 tokens |
| Checkpoint interval | Every 300 steps |
| Summary interval | Every 100 steps |
| TPU use | Disabled (`use_tpu=False`) |

The total training steps are computed from the number of training features, batch size, and epoch
count. The notebook trains a `tf.estimator.Estimator` with `estimator.train()` and evaluates it on
the validation features with `estimator.evaluate()`. Its evaluation function requests accuracy and
true/false positive/negative metrics. Since this is a nine-class task, those aggregate confusion
metrics are not a substitute for per-class precision, recall, F1, and a confusion matrix; the
notebook does not show those richer diagnostics.

### 6. Test prediction and artifacts

The prediction helper tokenizes test texts using the same BERT feature conversion and returns each
text's predicted class index, probability output, and mapped class name. Estimator checkpoints and
model files are written under the configured `OUTPUT_DIR`. The notebook does not show explicit
export to a versioned model registry or a deployment artifact contract.

## Technical Components

| Concern | Implementation in the notebook |
|---|---|
| Deep-learning framework | TensorFlow `tf.estimator` |
| Pretrained encoder | TensorFlow Hub BERT `bert_uncased_L-12_H-768_A-12/1` |
| BERT utilities | Google's `bert` package: `run_classifier`, `optimization`, `tokenization` |
| Input tokenization | BERT `FullTokenizer` with the Hub model vocabulary; uncased/lowercase |
| Tabular data and splits | pandas; scikit-learn `LabelEncoder` and `train_test_split` |
| Text cleanup | BeautifulSoup + Python regular expressions |
| Data/artifact storage | Azure Data Lake mounted in Databricks / DBFS |
| Credential access | Databricks secret scope `databricks-akv` |

## Label Mapping Caveat

The notebook's introductory text documents the nine labels with a specific numeric ordering, and
the prediction helper maps indices using that same listed order. However, training labels are
generated by `LabelEncoder`, which assigns integer values according to its own class ordering
(typically sorted class names). The code does not explicitly bind the encoder's `classes_` order to
the hard-coded prediction list. **Before reusing these artifacts, verify that the encoder's actual
class order matches the prediction mapping**; otherwise, displayed emotion names can be mismatched
to the model's predicted indices.

## Scope and Limitations

- This notebook trains the nine-class **Feelings/emotion** model; it is not the active
	three-class positive/neutral/negative sentiment model.
- The notebook is under `-deprecated-`; repository presence does not establish that it is still
	scheduled, maintained, or used to produce current model artifacts.
- It reads a path-specific labeled CSV and saves to a path-specific DBFS directory, so running it
	requires the corresponding data, permissions, secrets, and legacy TensorFlow/BERT dependencies.
- The split is not stratified, only 4% of the data is reserved for the test split, and the shown
	evaluation does not report class-wise metrics.
- Text cleaning removes non-ASCII characters and most punctuation; this can remove meaningful
	symbols or language-specific content and should be reviewed before retraining on broader data.
- No seed is set for TensorFlow model initialization/training, and no model registry, deployment,
	drift monitoring, or reproducibility manifest is shown.

## When Interpreting or Reusing It

Treat this as a historical example of transfer learning and multi-class text classification, not as
a ready-to-run production recipe. Before retraining, validate the label-index mapping, class balance,
train/validation/test methodology, preprocessing effects, per-class metrics, dependency compatibility,
and artifact versioning. Keep the nine-class emotion taxonomy distinct from any sentiment labels
used elsewhere in AIM+.



Code ref:
```

# Databricks notebook source
# MAGIC %md #### Predicting Feelings (Anticipation: 0, Concerned: 1, Descriptive: 2, Dissatisfied: 3, Frustrated: 4, Impressed: 5, Inquisitive: 6, Satisfied: 7, Skeptical: 8) 
# MAGIC 
# MAGIC With BERT IN Tensorflow
# MAGIC Bidirectional Encoder Representations from Transformers or BERT for short is a very popular NLP model from Google known for producing state-of-the-art results in a wide variety of NLP tasks.

# COMMAND ----------

# MAGIC %md ## Importing Necessary Libraries

# COMMAND ----------

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
from bs4 import BeautifulSoup
import re

print("tensorflow version : ", tf.__version__)
print("tensorflow_hub version : ", hub.__version__)

# COMMAND ----------

#Importing BERT modules
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

# COMMAND ----------

# MAGIC %md ##Setting The Output Directory
# MAGIC ---
# MAGIC While fine-tuning the model, we will save the training checkpoints and the model in an output directory so that we can use the trained model for our predictions later.
# MAGIC 
# MAGIC The following code block sets an output directory :

# COMMAND ----------

# Set the output directory for saving model file
OUTPUT_DIR = '/dbfs/mnt/rdmiblob/Saumya/Feelings/outputs_tf'

#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False #@param {type:"boolean"}

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    pass

tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# COMMAND ----------

# MAGIC %md ##Loading The Data
# MAGIC ---
# MAGIC We will now load the data from rdmiblob storage and will also split the training set in to training validation and test sets.

# COMMAND ----------

blobname = "rdmiblob"
storageaccount = dbutils.secrets.get(scope = "databricks-akv", key = "storage-account-name")
mountname = "/rdmiblob"

configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope = "databricks-akv", key = "service-principal-id"),
           "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope = "databricks-akv", key = "service-principal-secret"),
           "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/63982aff-fb6c-4c22-973b-70e4acfb63e6/oauth2/token"}

source = "abfss://" + blobname + "@" + storageaccount + ".dfs.core.windows.net"
mount = "/mnt" + mountname

try:
  dbutils.fs.mount(
    source = source,
    mount_point = mount,
    extra_configs = configs)
  #dbutils.notebook.exit("Ok")
except Exception as e:
  
  pass
  #dbutils.notebook.exit(str(e))

# COMMAND ----------

path = "/dbfs" + mount + "/Saumya/Feelings/"
print(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Function to do clean up on training data

# COMMAND ----------

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    #text = BeautifulSoup(text, "lxml").get_text()
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Keeping only letters
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    # Removing additional whitespaces
    text = re.sub(r" +", ' ', text)
    
    return text


# COMMAND ----------

train = pd.read_csv(path +"data_feelings.csv")
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
train['labels'] = labelencoder.fit_transform(train['emotion'])
train.head()
train['text'] = train['text'].apply(lambda x: clean_text(x))

# COMMAND ----------

from sklearn.model_selection import train_test_split
train, val =  train_test_split(train, test_size = 0.2, random_state = 100)
val, test = train_test_split(val, test_size = 0.2, random_state = 100)

# COMMAND ----------

train_df = train[['text','labels']]
eval_df = val[['text','labels']]
test_df = test[['text','labels']]

# COMMAND ----------

#Training set sample
train.head()

# COMMAND ----------

#Test set sample
test.head()

# COMMAND ----------

print("Training Set Shape :", train.shape)
print("Validation Set Shape :", val.shape)
print("Test Set Shape :", test.shape)

# COMMAND ----------

#val.to_csv(path + 'data_val.csv', index = False)

# COMMAND ----------

#Features in the dataset
train.columns

# COMMAND ----------

#unique classes
train['labels'].unique()

# COMMAND ----------

#Distribution of classes
train['emotion'].value_counts().plot(kind = 'bar')

# COMMAND ----------

DATA_COLUMN = 'text'
LABEL_COLUMN = 'labels'
# The list containing all the classes (train['SECTION'].unique())
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# COMMAND ----------

# MAGIC %md ## Data Preprocessing
# MAGIC 
# MAGIC BERT model accept only a specific type of input and the datasets are usually structured to have the following four features:
# MAGIC 
# MAGIC * guid : A unique id that represents an observation.
# MAGIC * text_a : The text we need to classify into given categories
# MAGIC * text_b: It is used when we're training a model to understand the relationship between sentences and it does not apply for classification problems.
# MAGIC * label: It consists of the labels or classes or categories that a given text belongs to.
# MAGIC  
# MAGIC In our dataset we have text_a and label. The following code block will create objects for each of the above mentioned features for all the records in our dataset using the InputExample class provided in the BERT library.

# COMMAND ----------

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

val_InputExamples = val.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

# COMMAND ----------

train_InputExamples

# COMMAND ----------

print("Row 0 - guid of training set : ", train_InputExamples.iloc[1].guid)
print("\n__________\nRow 0 - text_a of training set : ", train_InputExamples.iloc[1].text_a)
print("\n__________\nRow 0 - text_b of training set : ", train_InputExamples.iloc[1].text_b)
print("\n__________\nRow 0 - label of training set : ", train_InputExamples.iloc[1].label)

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we will use the ```bert_uncased_L-12_H-768_A-12/1``` model.
# MAGIC We will be using the vocab.txt file in the model to map the words in the dataset to indexes. Also, the loaded BERT model is trained on uncased/lowercase data and hence the data we feed to train the model should also be of lowercase.
# MAGIC 
# MAGIC The following code block loads the pre-trained BERT model and initializers a tokenizer object for tokenizing the texts.

# COMMAND ----------

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

# COMMAND ----------

#Here is what the tokenised sample of the first training set observation looks like
print(tokenizer.tokenize(train_InputExamples.iloc[1].text_a))

# COMMAND ----------

# MAGIC %md We will now format out text in to input features which the BERT model expects. We will also set a sequence length which will be the length of the input features.

# COMMAND ----------

# We'll set sequences to be at most 512 tokens long.
MAX_SEQ_LENGTH = 40

# Convert our train and validation features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

val_features = bert.run_classifier.convert_examples_to_features(val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# COMMAND ----------

#Example on first observation in the training set
print("Sentence : ", train_InputExamples.iloc[1].text_a)
print("-"*30)
print("Tokens : ", tokenizer.tokenize(train_InputExamples.iloc[1].text_a))
print("-"*30)
print("Input IDs : ", train_features[1].input_ids)
print("-"*30)
print("Input Masks : ", train_features[1].input_mask)
print("-"*30)
print("Segment IDs : ", train_features[1].segment_ids)

# COMMAND ----------

# MAGIC %md ##Creating A Multi-Class Classifier Model

# COMMAND ----------

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  
  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  sequence_output = tf.identity(bert_outputs["sequence_output"], name="sequence_output")
  print(bert_inputs)
  print(bert_outputs)
  print(sequence_output)

  model_outputs = dict(
     sequence_output=sequence_output)
  
  
  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# COMMAND ----------

#A function that adapts our model to work for training, evaluation, and prediction.
# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        
        return {
            "eval_accuracy": accuracy,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
            }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


# COMMAND ----------

# Compute train and warmup steps from batch size
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5.0
# Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100

# Compute train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

# COMMAND ----------

#Initializing the model and the estimator
model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

# COMMAND ----------

# MAGIC %md We will now create an input builder function that takes our training feature set (`train_features`) and produces a generator. This is a pretty standard design pattern for working with Tensorflow Estimators.

# COMMAND ----------

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

# Create an input function for validating. drop_remainder = True for using TPUs.
val_input_fn = run_classifier.input_fn_builder(
    features=val_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

# COMMAND ----------

# MAGIC %md ##Training & Evaluating

# COMMAND ----------

#Training the model
print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

# COMMAND ----------

#Evaluating the model with Validation set
estimator.evaluate(input_fn=val_input_fn, steps=None)

# COMMAND ----------

# MAGIC %md ##Predicting For Test Set

# COMMAND ----------

"""Anticipation: 0
Concerned: 1
Descriptive: 2
Dissatisfied: 3
Frustrated: 4
Impressed: 5
Inquisitive: 6
Satisfied: 7
Skeptical: 8"""

# A method to get predictions
def getPrediction(in_sentences):
  #A list to map the actual labels to the predictions
  labels = ['Anticipation','Concerned','Descriptive','Dissatisfied','Frustrated','Impressed','Inquisitive','Satisfied','Skeptical']

  #Transforming the test data into BERT accepted form
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] 
  
  #Creating input features for Test data
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

  #Predicting the classes 
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'],prediction['labels'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

# COMMAND ----------

pred_sentences = list(test['text'])

# COMMAND ----------

predictions = getPrediction(pred_sentences)

# COMMAND ----------

enc_labels = []
act_labels = []
for i in range(len(predictions)):
  enc_labels.append(predictions[i][2])
  act_labels.append(predictions[i][3])

# COMMAND ----------

pd.DataFrame(enc_labels, columns = ['emotion']).to_csv(path + 'results_tf.csv', index = False)

# COMMAND ----------

test.to_csv(path + 'data_test_tf.csv', index = False)

# COMMAND ----------

# MAGIC %md ## Random Tester

# COMMAND ----------

#Classifying random sentences
tests = getPrediction(['This Drug could have been way better',
                       'I am not happy with supply chain from GSK',
                       'That HBO TV series is really good',
                       'Tempaerature stability excursion',
                       "I would be wondering if you don't get good results"
                       ])

tests

```
