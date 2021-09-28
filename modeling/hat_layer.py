import tensorflow as tf

from modeling import attention_utils


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = attention_utils.MultiHeadAttention(d_model, num_heads)
    self.ffn = attention_utils.point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2


class HATEncoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(HATEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = attention_utils.positional_encoding(
      maximum_position_encoding, self.d_model
    )

    self.enc_layers = [
      EncoderLayer(d_model, num_heads, dff, rate) 
      for _ in range(num_layers)
    ]

    self.hat_layer = EncoderLayer(d_model, num_heads, dff, rate)

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask, bos_mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    bos_mask = tf.expand_dims(bos_mask, axis=-1)
    x *= tf.cast(bos_mask, dtype=tf.float32)
    hat_x = self.hat_layer(x, training, mask)

    return x, hat_x  # (batch_size, input_seq_len, d_model) x 2


class HATDecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(HATDecoderLayer, self).__init__()

    self.mha1 = attention_utils.MultiHeadAttention(d_model, num_heads)
    self.mha2 = attention_utils.MultiHeadAttention(d_model, num_heads)
    self.mha3 = attention_utils.MultiHeadAttention(d_model, num_heads)

    self.ffn = attention_utils.point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    self.dropout4 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, hat_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    attn3, attn_weights_block3 = self.mha2(hat_output, hat_output, out2, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn3 = self.dropout3(attn3, training=training)
    out3 = self.layernorm3(attn3 + out2)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out3)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out4 = self.layernorm4(ffn_output + out3)  # (batch_size, target_seq_len, d_model)
    
    return out4, attn_weights_block1, attn_weights_block2, attn_weights_block3


class HATDecoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(HATDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = attention_utils.positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [HATDecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, hat_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2, block3 = self.dec_layers[i](x, enc_output, hat_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
      attention_weights['decoder_layer{}_block3'.format(i+1)] = block3

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights
