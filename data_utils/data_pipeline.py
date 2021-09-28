import json
import collections
import tensorflow as tf
import sentencepiece as spm
from tqdm import tqdm

from data_utils import span_masking
from data_utils import tokenization


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, doc, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      doc: list. The untokenized text of the first sequence.
      label: (Optional) list. The label of the example.
        This should be specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.doc = doc
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               label_ids,
               bos_mask,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.label_ids = label_ids
    self.bos_mask = bos_mask
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Data converter for pre-training/fine-tuning data sets."""

  def __init__(self):
    super(DataProcessor, self).__init__()

  def get_train_examples(self, json_file, is_pretraining=False):
    """Makes train set from two files: input and label"""
    with open(json_file, 'r', encoding='utf-8') as f:
      d = json.load(f)
    
    label_key = "inputs" if is_pretraining else "outputs"
    examples = [InputExample(guid=k, doc=v["inputs"], label=v[label_key]) for k, v in d.items()]
    return examples


def convert_single_example(ex_index, example, input_length, output_length, encoder_tokenizer, decoder_tokenizer, is_pretraining=False):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * input_length,
        label_ids=[0] * output_length,
        bos_mask=[0] * output_length,
        is_real_example=False)

  input_tokens = ['[CLS]']
  for segment in example.doc:
    new_tokens = encoder_tokenizer.tokenize(segment.strip())
    input_tokens += new_tokens
    input_tokens.append('[SEP]')

  if is_pretraining:
    input_tokens = span_masking.span_masking(input_tokens)

  if len(input_tokens) > input_length:
    input_tokens = input_tokens[:input_length]

  bos_mask = []
  for token in input_tokens:
    is_bos = 1 if token in ['[CLS]', '[SEP]'] else 0
    bos_mask.append(is_bos)
  
  label_tokens = ['[CLS]']
  for segment in example.label:
    new_tokens = decoder_tokenizer.tokenize(segment.strip())
    label_tokens += new_tokens
    label_tokens.append('[SEP]')

  if len(label_tokens) > output_length:
    label_tokens = label_tokens[:output_length]
  
  input_ids = encoder_tokenizer.convert_tokens_to_ids(input_tokens)

  # Zero-pad up to the sequence length.
  while len(input_ids) < input_length:
    input_ids.append(0)
    bos_mask.append(0)

  assert len(input_ids) == input_length
  assert len(bos_mask) == input_length

  label_ids = decoder_tokenizer.convert_tokens_to_ids(label_tokens)

  # Zero-pad up to the sequence length.
  while len(label_ids) < output_length:
    label_ids.append(0)

  assert len(label_ids) == output_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("input_tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in input_tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("label_tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in label_tokens]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    tf.logging.info("bos_mask: %s" % " ".join([str(x) for x in bos_mask]))

  feature = InputFeatures(
      input_ids=input_ids,
      label_ids=label_ids,
      bos_mask=bos_mask,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, input_length, output_length, encoder_tokenizer, decoder_tokenizer, output_file, is_pretraining=False):
  """Convert a set of `InputExample`s to a TFRecord file."""

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in tqdm(enumerate(examples)):
    feature = convert_single_example(
      ex_index, example, input_length, output_length, encoder_tokenizer, decoder_tokenizer, is_pretraining=is_pretraining
    )
  
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["bos_mask"] = create_int_feature(feature.bos_mask)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
  
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, input_length, output_length, is_training,
                                drop_remainder, batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.int64
  name_to_features = {
      "input_ids": tf.FixedLenFeature([input_length], tf.int64),
      "label_ids": tf.FixedLenFeature([output_length], tf.int64),
      "bos_mask": tf.FixedLenFeature([input_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  def input_fn(params):
    """The actual input function."""
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn