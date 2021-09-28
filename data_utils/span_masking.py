import random

def get_prob_of_span_length(n, p):
  prob_of_span_length = [0.0]
  tmp, total = 1.0, 0.0
  for _ in range(n):
    cur_prob = tmp*p
    prob_of_span_length.append(cur_prob)

    total += cur_prob
    tmp *= (1-p)

  prob_of_span_length[0] = 1-total
  return prob_of_span_length

def pred_span_length(prob_of_span_length):
  p = random.random()
  for i, prob in enumerate(prob_of_span_length):
    if p  < prob:
      return i
    p -= prob
  return -1

def span_masking(tokens, coverage=0.15, span_p=0.2, lmax=10):
  if (len(tokens) - 2) * span_p < 2:
    return tokens

  prob_of_span_length = get_prob_of_span_length(lmax, span_p)

  splitted_tokens = []
  for token in tokens[1:-1]:
    if token.startswith('_') or token == '[SEP]':
      splitted_tokens.append([token])
    elif len(splitted_tokens) > 0:
      splitted_tokens[-1].append(token)

  word_len = len(splitted_tokens)
  masked_tokens = ['[CLS]']
  i = 0
  while i < word_len:
    p = random.random()
    if p < span_p and masked_tokens[-1] != "[MASK]":
      if splitted_tokens[i][0] != '[SEP]':
        masked_tokens.append("[MASK]")
      span_len = pred_span_length(prob_of_span_length)
      if span_len < 0:
        span_len = 0
      elif span_len > word_len * coverage:
        span_len = int(word_len * coverage)
      for _ in range(span_len):
        if i >= word_len:
          break
        if splitted_tokens[i][0] == '[SEP]':
          break
        i += 1
    else:
      masked_tokens += splitted_tokens[i]
      i += 1
  masked_tokens.append('[SEP]')
  return masked_tokens


if __name__ == "__main__":
  tokens = [
    '[CLS]', '_Hello', ',', '_my', '_name', '_is', '_Jin', 'hy', 'uk', '_Yang', '.', '[SEP]',
    '_Wel', 'come', '_to', '_At', 'las', 'labs', '_open', '_source', '!', '[SEP]',
    '_If', '_source', '_code', '_is', '_not', '_work', 'ing', '_please', '_con', 'tact', '_us', '_any', 'time', '.', '[SEP]'
  ]
  masked_tokens = span_masking(tokens)

  print("input_tokens:")
  print(tokens)
  print("output_tokens:")
  print(masked_tokens)