#!/bin/bash
# Train the skipthought vector model using text data

# The path to the data directory
DATA="../data/skipthought/"

# Download the training data
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget -P $DATA http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

