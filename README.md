#### Evaluation of ConvKB on FB15k after fixing the bug.

The results of ConvKB on FB15k are in checkpoints/model-200.eval.0.txt:

8725015.0 3213.2220056354204 6145.0
3956645.0 6726.795358734404 11121.0

which represents:

The MR, MRR and HITS10 of **head_results** are:

8725015.0 / 20466 = **426**, 3213.2220056354204 / 20466 * 100 = **15.7**, 6145.0 / 20466 * 100 = **30.0**

And the MR, MRR and HITS10 of **tail_results** are:

3956645.0 / 20466 = **193**, 6726.795358734404 / 20466 * 100 = **32.8**, 11121.0 / 20466 * 100 = **54.3**

##### So, the overall results are:

**MR = 309, MRR = 24.25, HITS10 = 42.16**



Our eval.py and model.py would fix the bug.

We also provide the script that training and evaluating ConvKB on FB15k.

We also open a bug-fixing pull request (https://github.com/daiquocnguyen/ConvKB/pull/3), which is closed by the ConvKB author.
