# vanilla-rnn-sentiment-analysis
To better understand how RNNs actually work, I implemented the math behind rnn feedforward and backpropogation using numpy. This project takes in example sentences such "This was very sad earlier", which are all tagged by their sentiment (True meaning happy and False meaning sad), and attempts to predict the sentiment of sentences that are not part of its training set. With simple sentences, it can achieve upwards of 90% accuracy. 

## problems:
- It has a difficult time understanding more complex sentences. This is likely due to my computers inability to process large enough datasets to train the model properly. 
- Currently, I do not have a good mathematical solution to the exploding or vanishing gradient problem. 
