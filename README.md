# supervised-classification-of-ssh-publications-into-disciplinary-categories
In this project we develop and evaluate a machine learning module that automatically classifies scientific documents into fine grained disciplinary categories.

This repository contains the code used for merging the data and performing the analyses as described in [paper].

# TO DO
- [ ] Make subdirectories in repo for all csv's and excels
- [ ] More explanations for each step/phase in notebook --> make it readable as a standalone publication
- [ ] Add actual machine learning part (PART 4)
- [ ] Double check train-test-split and crossvalidation
- [ ] Cleanup code!
- [ ] Add iterators in stead of repetitions of code
- [ ] Add random seeds and states everywhere! (the magic number is 56)
- [ ] Add citations to papers describing different algorithms used

## PART 1 - Data wrangling
 1. Data merging
 2. Data cleaning and relabeling
 3. Descriptive statistics
    - Number of records per disciplinary category
    - Number of labels per record
      - Per discipline
      - On level 3 
      - On level 4

## PART 2 - Evaluation of inter-indexer consistency
 1. Random sampling procedure
 2. Evaluation per set
    - On level 3
    - On level 4

## PART 3 - Data pre-processing
 1. Tokenization
    - Lemmas
    - Nouns
    - Noun phrases
 2. Count- and TF-IDF transformations

## PART 4 - Supervised classification
 1. Train-test-split
 2. Multinomial Naive Bayes
    - Parameter search: Randomized grid search
    - Evaluate predictions on holdout set
 3. Gradient Boosting
    - Parameter search: Randomized grid search
    - Evaluate predictions on holdout set
