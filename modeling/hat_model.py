import re
import collections
import tensorflow as tf
from absl import logging

from modeling import attention_utils
from modeling import hat_layer
from optimizer import optimization


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, num_of_group=0):
  """Compute the union of the current variables and checkpoint variables."""
  
  initialized_variable_names = {}
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
  init_vars = tf.train.list_variables(init_checkpoint)
  init_vars_name = [name for (name, _) in init_vars]

  assignment_map = [collections.OrderedDict() for _ in range(num_of_group)] if num_of_group > 0 else collections.OrderedDict()

  for name in name_to_variable:
    if name in init_vars_name:
      tvar_name = name
    elif (re.sub(r"/group_\d+/", "/group_0/",
                 six.ensure_str(name)) in init_vars_name and
          num_of_group > 1):
      tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
    elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
          in init_vars_name and num_of_group > 1):
      tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
    elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
          in init_vars_name and num_of_group > 1):
      tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                         six.ensure_str(name))
    else:
      tf.logging.info("name %s does not get matched", name)
      continue
    tf.logging.info("name %s match to %s", name, tvar_name)
    if num_of_group > 0:
      group_matched = False
      for gid in range(1, num_of_group):
        if (("/group_" + str(gid) + "/" in name) or
            ("/ffn_" + str(gid) + "/" in name) or
            ("/attention_" + str(gid) + "/" in name)):
          group_matched = True
          tf.logging.info("%s belongs to %dth", name, gid)
          assignment_map[gid][tvar_name] = name
      if not group_matched:
        assignment_map[0][tvar_name] = name
    else:
      assignment_map[tvar_name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[six.ensure_str(name) + ":0"] = 1

  return (assignment_map, initialized_variable_names)


class HATTransformer(tf.keras.Model):
  def __init__(self, num_encoding_layers, num_decoding_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(HATTransformer, self).__init__()

    self.encoder = hat_layer.HATEncoder(
        num_encoding_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
    )

    self.decoder = hat_layer.HATDecoder(
        num_decoding_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
    )

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, tar, training, enc_padding_mask, enc_bos_mask,
           look_ahead_mask, dec_padding_mask):

    enc_output, hat_output = self.encoder(inp, training, enc_padding_mask, enc_bos_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, hat_output, training, look_ahead_mask, dec_padding_mask
    )

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights


def loss_transformer(logits, labels):
  mask = tf.cast(
      tf.math.logical_not(tf.math.equal(labels, 0)),
      dtype=tf.float32
  )

  crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
  )
  loss = tf.reduce_sum(crossentropy(labels, logits) * mask)
  
  total_size = tf.reduce_sum(mask) + 1e-12
  loss /= total_size
  return loss


def softmax_layer(logits):
  # predict not mask we could filtered it in the prediction part.
  probabilities = tf.math.softmax(logits, axis=-1)
  predict = tf.math.argmax(probabilities, axis=-1)
  return predict


def create_model(hat_config, is_training, input_ids, label_ids, bos_mask):
  """Creates HAT model."""

  # decoder inference setting
  tar_inp, tar_real = label_ids[:, :-1], label_ids[:, 1:]

  # create mask
  enc_padding_mask, combined_mask, dec_padding_mask = attention_utils.create_masks(
    input_ids, tar_inp
  )

  # HAT Transformer
  transformer = HATTransformer(
      num_encoding_layers=hat_config["num_of_encoding_layers"],
      num_decoding_layers=hat_config["num_of_decoding_layers"],
      d_model=hat_config["hidden_size"],
      num_heads=hat_config["attention_heads"],
      dff=hat_config["feed_forward_size"],
      input_vocab_size=hat_config["encoder_vocab_size"],
      target_vocab_size=hat_config["decoder_vocab_size"],
      pe_input=hat_config["input_length"],
      pe_target=hat_config["output_length"]-1
  )
  
  logits, _ = transformer(
      input_ids, tar_inp,
      training=is_training,
      enc_padding_mask=enc_padding_mask,
      enc_bos_mask=bos_mask,
      look_ahead_mask=combined_mask,
      dec_padding_mask=dec_padding_mask
  )

  # softmax layer and loss function
  predicts = softmax_layer(logits)
  total_loss = loss_transformer(logits, tar_real)
  
  return total_loss, logits, predicts


def model_fn_builder(hat_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
  def model_fn(features, mode, params):
    
    # recognize input features
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    input_ids = features["input_ids"]
    label_ids = features["label_ids"]
    bos_mask = features["bos_mask"]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # load model
    total_loss, logits, predicts = create_model(
        hat_config, is_training, input_ids, label_ids, bos_mask
    )
    tvars = tf.trainable_variables()
    initialized_variable_names=None

    # checkpoint
    if init_checkpoint:
      assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    
    # train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op
      )

    # validation mode
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss
      )

    # test mode
    else:
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=logits
      )
    return output_spec

  return model_fn
