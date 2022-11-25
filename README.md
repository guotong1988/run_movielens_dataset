# Run Movielens Dataset
Get AUC 0.794 at Movielens 20M dataset

# Requirements

python3 

tensorflow1.14 or 1.15

# Reproduce
step0, download dataset from https://download.csdn.net/download/guotong1988/85505311

step1, run `data/preprocess.py`

step2, run `train.py`

# Results

| **Method**   | **AUC** |  
| ----------- | ----------- | 
| [Deep Interest Network](https://paperswithcode.com/paper/deep-interest-network-for-click-through-rate)    | 0.73           | 
| [Wide & Deep](https://paperswithcode.com/paper/wide-deep-learning-for-recommender-systems)|  0.73 |
| Our MLP    | 0.792           | 
| Our transformer |   0.794                  | 

# Reasonable

Our method focus on the data preprocess step for the Movielens dataset. In detail, we get the top-15 tags for each user and top-15 tags for each item/movie. We view each tag as an id.
