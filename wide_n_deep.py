# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 500, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

COLUMNS = ["userid", "movieid", "rating"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["userid", "movieid"]
CONTINUOUS_COLUMNS = [ "rating"]



def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  rating = tf.contrib.layers.real_valued_column("rating")
  #timestamp = tf.contrib.layers.sparse_column_with_hash_bucket(
  #    "timestamp", hash_bucket_size=100)
  # Continuous base columns.
  userid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="userid",hash_bucket_size=1000)

  movieid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="movieid",hash_bucket_size=1000)

  # Transformations.

  # Wide columns and deep columns.
  wide_columns = [userid,movieid]
  deep_columns = [ tf.contrib.layers.embedding_column(userid,dimension=8),
                   tf.contrib.layers.embedding_column(movieid,dimension=8),
                   rating ]

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns
                                        )
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50],
                                       )
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50]
        )
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].astype(str).values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
  train_file_name = "./train.csv"
  test_file_name = "./test.csv"
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      dtype={
      "userid": str,
      "movieid":str,
      "rating": int,
      },
      names=COLUMNS,
      skipinitialspace=True)
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      dtype={
     "userid": str,
      "movieid": str,
      "rating": int,
      },
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1)
  df_train[LABEL_COLUMN] = df_train["rating"].apply(lambda x: 0 if(x<3) else 1)
  df_test[LABEL_COLUMN] = df_test["rating"].apply(lambda x:0 if(x<3) else 1)
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)
  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
def main(_):
  train_and_eval()
if __name__ == "__main__":
  tf.app.run()
