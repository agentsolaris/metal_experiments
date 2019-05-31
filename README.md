In this repository, we currently have three files:
* MMTL_BERT.py : This performs a MMTL training on two datasets over BERT for a certain amount of epochs, before printing the result of this training.
* MMTL_BERT_Eval.py : As above, except the training is done one epoch at a time in a loop, and the results are printed per epoch.
* make_test_datasets.py : Creates modified datasets that can be used in the above experiments. It creates smaller datasets, by randomly sampling 5% of the original data, which is useful for running many experiments in quick succession. It also creates binary datasets, where only 2 classes are chosen to be included in the dataset, which can be useful to make the ML problem easier.
