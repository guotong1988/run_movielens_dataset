# Run Movielens Dataset
Get AUC 0.8 at Movielens 20M dataset

# Reproduce
step1, run `data/preprocess.py`

step2, run `train.py`

# Results

| **Method**   | **AUC** |  
| ----------- | ----------- | 
| [Deep Interest Network](https://paperswithcode.com/paper/deep-interest-network-for-click-through-rate)    | 0.73           | 
| [Wide & Deep](https://paperswithcode.com/paper/wide-deep-learning-for-recommender-systems)|  0.73 |
| Our MLP    | 0.79           | 
| Our transformer |                      | 

# Reasonable

Our method focus on the data preprocess step for the Movielens dataset. In detail, we get the top-15 tags for each user and top-15 tags for each item/movie. We view each tag as an id.
