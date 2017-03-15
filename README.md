# Skip Thought Sentence Vectors
An implementation of skip-thought vectors in Tensorflow

## Dependencies

This code is written in python. To use it you will need:
* Python 2.7
* Tensorflow 1.0
* NumPy
* SciPy
* scikit-learn
* NLTK 3

Exact versions are listed in requirements.txt

## Quick Start
The default training procedure is handled by `train.sh`.
Update the variables in `train.sh` for your local configuration an then run:

```sh
bash train.sh
```
## Training
The following steps are implimentent in `train.sh`. They are included here for completeness

You will first need to download the model files and word embeddings. The embedding files (utable and btable) are quite large (>2GB) so make sure there is enough space available. The encoder vocabulary can be found in dictionary.txt.

```sh
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

## Usage
Once the Tensorflow model has bee trained, it can be used to generate sentence vectors:

```python3
from skipthought import SkipthoughtModel
from skipthought.data_utils import TextData
from skipthought.utils import seq2seq_triples_data_iterator

model = SkipthoughtModel(...)

td = TextData("path/to/data")
lines = td.dataset

prev, curr, next = td.make_triples(lines)
it = td.triples_data_iterator(prev, curr, next, td.max_len, batch_size)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    for enc_inp, prev_inp, prev_targ, next_inp, next_targ in it:
        ....

```