# seq2seq: Encoder-Decoder models
## Requirement
+ Python 3
+ Keras

## Work: implemented seq2seq models without using recurrent shop
+ Simple Encoder-Decoder
+ Attentative Encoder-Decoder

Those models only depend on Keras Functional API.
If you want to know more details, see models.py.

## Warning! This repository is outdated!

Although some of you starred this repository (thx!), here is almost outdated.
And I also found mistakes in coding where is the input for decoder.
My decoder takes input as a sentence, but correct decoder should take input as a word.
In other word, decoder predict (w_t) given (w_t-1, hidden_t-1), not given ([w_o; w_1; ...; w_t-1], hidden_t-1) as I coded.
However, except that, some codes here are still helpful for learner of Keras and Natural Language Processing with Deep Learning.

By the way, I swiched to use [Pytorch](http://pytorch.org) insted of using Keras for some reasons.
Pytorch is also very cool framework, which is easy to construst heavy deep models.
[This repository](https://github.com/tatsuokun/pytorch_seq2seq) contains pytorch implementation of Seq2Seq that is the same to this Keras implementation.
In addition to Attention model, I implemented [Sentinel, CVPR 2017](https://arxiv.org/pdf/1612.01887.pdf), [Selective Gate, ACL 2017](https://arxiv.org/pdf/1704.07073.pdf) and [Gumbel Softmax, ICML 2017](https://arxiv.org/pdf/1611.01144.pdf) in pytorch.
If you are interested in latest deep models, visit [here](https://github.com/tatsuokun/pytorch_seq2seq).
