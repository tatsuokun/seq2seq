from collections import Counter


def vocab(path, vocab_size):
    with open(path, mode="r") as f1:
        token = []
        for line in f1:
            word = line.strip().split(" ") + ["</s>"]
            for w in word:
                token.append(w)
        vocab = Counter(token).most_common(vocab_size)
        vocab, _ = zip(*vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(vocab))
        idx_word = dict((value, key) for key, value in word_idx.items())
        return word_idx, idx_word


def word2idx(word_idx, sentence):
    unk = len(word_idx) + 1
    return [unk if word not in word_idx else word_idx[word] for word in sentence]


def sentence2idx(path, vocab_size=False, word_idx=False, train=False):
    # convert sentence words into idx using given word_idx dictionary
    if not vocab_size and word_idx:
        with open(path, mode="r") as f:
            if train:
                sentences = [["</s>"]+word2idx(word_idx, line.strip().split()+["</s>"]) for line in f]
            else:
                sentences = [word2idx(word_idx, line.strip().split()+["</s>"]) for line in f]
        return sentences, False, False

    elif vocab_size:
        word_idx, idx_word = vocab(path, vocab_size)
        with open(path, mode="r") as f:
            if train:
                sentences = [["</s>"]+line.strip().split() for line in f]
            else:
                sentences = [line.strip().split()+["</s>"] for line in f]
        sentences = [word2idx(word_idx, sentence) for sentence in sentences]
        return sentences, word_idx, idx_word

    else:
        raise TypeError
