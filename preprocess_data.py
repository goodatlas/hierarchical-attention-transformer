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

from data_utils import tokenization
from data_utils import data_pipeline

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "hat_config", None,
    "JSON file for HAT model configuration"
)

flags.DEFINE_string(
    "trainset_file", None,
    "JSON file of train set."
)

flags.DEFINE_string(
    "intermediate_dir", None,
    "The directory to store intermediate TFRecords files"
)

flags.DEFINE_string(
    "encoder_vocab_file", None,
    "The vocabulary file for SentencePiece tokenization on Transformer encoder"
)

flags.DEFINE_string(
    "encoder_spm_model_file", None,
    "The model file for SentencePiece tokenization on Transformer encoder"
)

## Other parameters
flags.DEFINE_bool(
    "pretrain_mode", False,
    "If you want to run on pre-training mode, please set this parameter True"
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. "
    "Should be True for uncased models and False for cased models."
)

flags.DEFINE_bool(
    "share_tokenizer", False,
    "Set True if you want to use same tokenizer between encoder and decoder. "
    "Else, please set tokenizer information for decoder."
)

flags.DEFINE_string(
    "decoder_vocab_file", None,
    "The vocabulary file for SentencePiece tokenization on Transformer decoder"
)

flags.DEFINE_string(
    "decoder_spm_model_file", None,
    "The model file for SentencePiece tokenization on Transformer decoder"
)

def main(_):
    logging.set_verbosity(logging.INFO)
    
    processor = data_pipeline.DataProcessor()
    encoder_tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.encoder_vocab_file,
        do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.encoder_spm_model_file
    )
    if FLAGS.share_tokenizer:
        decoder_tokenizer = encoder_tokenizer
    else:
        decoder_tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.decoder_vocab_file,
            do_lower_case=FLAGS.do_lower_case,
            spm_model_file=FLAGS.decoder_spm_model_file
        )
    
    with open(FLAGS.hat_config, 'r', encoding='utf-8') as f:
        hat_config_dict = json.load(f)

    train_examples = processor.get_train_examples(
        FLAGS.trainset_file, is_pretraining=FLAGS.pretrain_mode
    )

    filename = "pretrain.json" if FLAGS.pretrain_mode else "train.json"
    statistic_file = os.path.join(FLAGS.intermediate_dir, filename)
    with open(statistic_file, 'w', encoding='utf-8') as fo:
        json.dump({"num_of_examples": len(train_examples)}, fo)
    
    filename = "pretrain.tf_record" if FLAGS.pretrain_mode else "train.tf_record"
    train_file = os.path.join(FLAGS.intermediate_dir, filename)
    data_pipeline.file_based_convert_examples_to_features(
        train_examples,
        hat_config_dict["input_length"], hat_config_dict["output_length"],
        encoder_tokenizer, decoder_tokenizer, train_file,
        is_pretraining=FLAGS.pretrain_mode
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("hat_config")
    flags.mark_flag_as_required("trainset_file")
    flags.mark_flag_as_required("intermediate_dir")
    flags.mark_flag_as_required("encoder_vocab_file")
    flags.mark_flag_as_required("encoder_spm_model_file")
    tf.app.run()
