import numpy as np

class LanguageModel:
    def __init__(self, text, word_limit=100):
        self.words = text.lower().split()[:word_limit]
        self.vocab = list(set(self.words))
        self.word2index = {word: i for i, word in enumerate(self.vocab)}
        self.index2word = {i: word for i, word in enumerate(self.vocab)}
        self.bigram = self.build_bigram()

    def build_bigram(self):
        bigram = np.zeros((len(self.vocab), len(self.vocab)))
        for i in range(len(self.words) - 1):
            current_word = self.words[i]
            next_word = self.words[i + 1]
            bigram[self.word2index[current_word]][self.word2index[next_word]] += 1
        bigram /= (np.sum(bigram, axis=1, keepdims=True) + 1e-10) 
        return bigram

    def predict_next_word(self, current_word):
        if current_word not in self.word2index:
            return None
        next_word_index = np.random.choice(range(len(self.vocab)), p=self.bigram[self.word2index[current_word]])
        return self.index2word[next_word_index]

with open('data.txt', 'r', encoding='utf-8') as file:
    data_text = file.read()

lm = LanguageModel(data_text)
print(lm.predict_next_word("to"))
