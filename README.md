## Evaluation of ConvKB on FB15k after fixing the bug

#### The bug in ConvKB evaluation

We found that the ConvKB model tends to assign the same distance score, namely, the bias in the model,  to many triplets. The reason is that the RELU activation function is used in the convolution layers, which tends to have very sparse output, i.e., the output of many neurons are zero. This brings a big problem in the evaluation.

For evaluation, given a query (h,r, ?), the goals is to identify the rank of the true positive triplets (h, r, t) among all the possible (h, r, t’) triplets. Since the scores of many triplets given by ConvKB equal to the same score (or the bias), the true positive triplets and many other false triplets are all ranked the first position at the same time. A reasonable solution would be to randomly pick a triplet among those triplets as the first ranked triplet, and so on. However,  we find that a specific ranking procedure is used by ConvKB, which tends to rank the true positive triplets in a high position. As a result, the performance evaluated in this way is really high, which is not true in reality.

#### How do we fix the bug

Our eval.py and model.py would fix the bug.

In model.py, we removed the constraint on the batch size:

```python
#from
self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
#to
self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
```

In eval.py, we removed the codes that create the duplicates of the correct triplets:

```python
#remove the following codes
while len(new_x_batch) % ((int(args.neg_ratio) + 1) * args.batch_size) != 0:
    new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
    new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)
```

We also provide the script that we used for training and evaluating ConvKB on FB15k.

#### Our results after fixing the bug

The results of ConvKB on FB15k are in checkpoints/model-200.eval.0.txt:

8725015.0 3213.2220056354204 6145.0

3956645.0 6726.795358734404 11121.0

which represent:

ConvKB MR, MRR and HITS10 of **head_results** are:

8725015.0 / 20466 = **426**, 3213.2220056354204 / 20466 * 100 = **15.7**, 6145.0 / 20466 * 100 = **30.0**

And ConvKB MR, MRR and HITS10 of **tail_results** are:

3956645.0 / 20466 = **193**, 6726.795358734404 / 20466 * 100 = **32.8**, 11121.0 / 20466 * 100 = **54.3**

**So, the overall results of ConvKB on FB15k are:**

**MR = 309, MRR = 24.25, HITS10 = 42.16**

