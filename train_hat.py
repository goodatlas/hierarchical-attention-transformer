from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pickle
import collections
import tensorflow as tf
import numpy as np
from absl import flags, logging

from data_utils import data_pipeline
from modeling import hat_model

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "hat_config", None,
    "JSON file for HAT model configuration"
)

flags.DEFINE_string(
    "trainset_file", None,
    "The pre-processed intermediate TFRecords file for training"
)

flags.DEFINE_string(
    "count_file", None,
    "The pre-processed intermediate JSON file for training"
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "export_model_dir", None,
    "The path to export model for serving"
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint. Both pre-trained model and fine-tuned model are possible.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("export_batch_size", 1, "Total batch size for exported model.")

flags.DEFINE_float("learning_rate", 5e-4, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

def main(_):
    logging.set_verbosity(logging.INFO)

    # multi-GPU training configuration
    dist_strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps, 
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        train_distribute=dist_strategy
    )

    # count steps
    with open(FLAGS.count_file, 'r', encoding='utf-8') as fi:
        num_of_examples = json.load(fi)["num_of_examples"]
    num_train_steps = int(
        num_of_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    # declare model
    with open(FLAGS.hat_config, 'r', encoding='utf-8') as fi:
        hat_config_dict = json.load(fi)

    model_fn = hat_model.model_fn_builder(
        hat_config=hat_config_dict,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={'batch_size': FLAGS.train_batch_size}
    )

    # train model
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", num_of_examples)
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = data_pipeline.file_based_input_fn_builder(
        input_file=train_file,
        input_length=hat_config_dict["input_length"],
        output_length=hat_config_dict["output_length"],
        is_training=True,
        drop_remainder=True,
        batch_size=FLAGS.train_batch_size)
    
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # export model
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        reciever_tensors = {
            "input_ids": tf.placeholder(
                dtype=tf.int64,
                shape=[FLAGS.export_batch_size, FLAGS.max_seq_length]
            ),
            "label_ids": tf.placeholder(
                dtype=tf.int64,
                shape=[FLAGS.export_batch_size, FLAGS.max_output_length]
            )
        }
        features = {
            "input_ids": reciever_tensors['input_ids'],
            "input_mask": 1 - tf.cast(tf.equal(reciever_tensors['input_ids'], 0), dtype=tf.int64),
            "segment_ids": tf.zeros(dtype=tf.int64,
                                    shape=[FLAGS.export_batch_size, FLAGS.max_seq_length]),
            'label_ids': reciever_tensors['label_ids'],
            'label_mask': 1 - tf.cast(tf.equal(reciever_tensors['label_ids'], 0), dtype=tf.int64)
        }
        return tf.estimator.export.ServingInputReceiver(features, reciever_tensors)
    
    estimator.export_savedmodel(FLAGS.export_model_dir, serving_input_receiver_fn, strip_default_attrs=True)


if __name__ == "__main__":
    flags.mark_flag_as_required("hat_config")
    flags.mark_flag_as_required("trainset_file")
    flags.mark_flag_as_required("count_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("export_model_dir")
    tf.app.run()
