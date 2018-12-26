'''
Sandbox for generative language models of sequences
'''

import argparse
from functools import reduce
import random

import numpy as np

from mwm.data import load_sonnets, load_bible

START_SYMBOL = '<START>'
END_SYMBOL = '<END>'
OOV_SYMBOL = '<OOV>'

MAX_SEQUENCE_LENGTH = 1000

class LM:
    def __init__(self):
        pass
    
    def generate(self, k):
        '''
        Sample k examples from this language model
        '''
        
        raise NotImplementedError(
            'Need to implement "{}"'.format(generate.__name__)
        )
    
    def log_probability(self, seq):
        '''
        Returns log-probability assigned to this sequence (base 2)
        '''
        
        raise NotImplementedError(
            'Need to implement "{}"'.format(likelihood.__name__)
        )
    
    def probability(self, seqs):
        '''
        Probability of dataset (examples independent).
        '''
        
        prob_per_seq = [2.**self.log_probability(seq) for seq in seqs]
        return reduce(lambda x, y: x * y, prob_per_seq, 1.)
    
    def perplexity_per_seq(self, seqs):
        '''
        Perplexity per example
        
        https://en.wikipedia.org/wiki/Perplexity
        '''
        
        log_prob_per_seq = [self.log_probability(seq) for seq in seqs]
        
        return 2.**(-sum(log_prob_per_seq) / len(seqs))
    
    def perplexity_per_token(self, seqs):
        '''
        Perplexity per token
        
        https://en.wikipedia.org/wiki/Perplexity
        '''
        
        log_prob_per_seq = [self.log_probability(seq) for seq in seqs]
        
        return 2.**(-sum(log_prob_per_seq) /
                    sum([len(seq) for seq in seqs]))


class MemorizerLM(LM):
    '''
    Very simple language model -- memorize the training set and treat
    each sequence separately.
    '''
    
    def fit(self, seqs):
        self._training_set = {}

        # count number of times each sequence occurs
        for seq in seqs:
            if tuple(seq) not in self._training_set:
                self._training_set[tuple(seq)] = 0
            self._training_set[tuple(seq)] += 1
        
        self._N = sum(self._training_set.values())
        self._invN = 1. / self._N
        
        # keep track of frequency for each sequence, for generation
        self._training_lst = sorted(list(self._training_set.keys()))
        self._training_wts = [self._training_set[k] * self._invN for k
                              in sorted(list(self._training_set.keys()))]
    
    def generate(self, k):
        synthetic_seqs = np.random.choice(self._training_lst,
                                          size=k,
                                          p=self._training_wts)

        synthetic_seqs = [[START_SYMBOL] + list(seq) + [END_SYMBOL]
                          for seq in synthetic_seqs]
        
        return synthetic_seqs
    
    def log_probability(self, seq):
        if tuple(seq) in self._training_set:
            return np.log2(self._training_set[tuple(seq)] / self._N)
        else:
            return float('-inf')
        

class UnigramLM(LM):
    def __init__(self, heldout_mass):
        self._vocab = {}
        self._log_prob_per_token = {}
        self._sorted_vocab = []
        self._vocab_weights = []

        self._heldout_mass = heldout_mass
        
        self._is_fit = False
    
    def fit(self, seqs):
        vocab_counts = {}
        self._vocab = {}
        
        N = 0
        vocab_idx = 0
        
        for seq in seqs:
            for token in seq + [END_SYMBOL]:
                if token not in vocab_counts:
                    vocab_counts[token] = 0.
                    self._vocab[token] = vocab_idx
                    vocab_idx += 1
                
                vocab_counts[token] += 1.
                N += 1
        
        self._log_prob_per_token = {t: np.log2((1. - self._heldout_mass) *
                                               vocab_counts[t] / N)
                                    for t in self._vocab}
        
        # reserve some probability mass for out-of-vocabulary tokens
        self._log_prob_per_token[OOV_SYMBOL] = np.log2(self._heldout_mass)
        self._vocab[OOV_SYMBOL] = vocab_idx
        
        self._sorted_vocab  = sorted(list(self._vocab.keys()))
        self._vocab_weights = [2.**self._log_prob_per_token[t]
                               for t in self._sorted_vocab]
        
        # hack to ignore start symbol when assigning sequence probability
        self._log_prob_per_token[START_SYMBOL] = 0.
        
        self._is_fit = True
    
    def generate(self, k):
        if not self._is_fit:
            raise Exception('UnigramLM not fit yet...')
        
        synthetic_seqs = []
        
        for _ in range(k):
            seq = [START_SYMBOL]
            
            # keep sampling token until we sample an end-of-sentence token
            while True and (len(seq) < MAX_SEQUENCE_LENGTH):
                sampled_token = np.random.choice(self._sorted_vocab,
                                                 size=1,
                                                 p=self._vocab_weights)[0]
                
                seq.append(sampled_token)
                
                if sampled_token == END_SYMBOL:
                    break
            
            synthetic_seqs.append(seq)
        
        return synthetic_seqs
    
    def log_probability(self, seq):
        log_prob = sum([self._log_prob_per_token[token]
                        if token in self._log_prob_per_token
                        else self._log_prob_per_token[OOV_SYMBOL]
                        for token
                        in [START_SYMBOL] + seq + [END_SYMBOL]])
        
        return log_prob


class NgramLM(LM):
    '''
    n-gram language model for arbitrary n
    https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    '''
    
    def __init__(self, n):
        self._n = n
    
    def fit(self, seqs):
        raise NotImplementedError(
            'Need to implement "{}"'.format(fit.__name__)
        )

    def generate(self, k):
        raise NotImplementedError(
            'Need to implement "{}"'.format(generate.__name__)
        )
    
    def probability(self, seq):
        raise NotImplementedError(
            'Need to implement "{}"'.format(probability.__name__)
        )


def main(dataset, num_examples_to_generate):
    # read examples, lists 
    if dataset == 'sonnets':
        seqs = load_sonnets()
    elif dataset == 'bible':
        seqs = load_bible()
    else:
        raise Exception('Unrecognized dataset: "{}"'.format(dataset))
    
    # hold out 50 sentences to estimate heldout likelihood
    random.shuffle(seqs)

    train_seqs = seqs[:-50]
    heldout_seqs = seqs[-50:]
    
    memorizer_lm = MemorizerLM()
    unigram_lm = UnigramLM(0.05) # hold out 5% of mass for out-of-vocabulary
    
    print('='*20 + ' Memorizer Language Model ' + '='*20)
    memorizer_lm.fit(train_seqs)
    print('Train (Token) Perplexity: {}'.format(
        memorizer_lm.perplexity_per_token(train_seqs))
    )
    print('Heldout (Token) Perplexity: {}'.format(
        memorizer_lm.perplexity_per_token(heldout_seqs))
    )
    
    synth_seqs = memorizer_lm.generate(num_examples_to_generate)
    print('==== Generated Examples ====')
    for seq in synth_seqs:
        print(' -> ' + ' '.join(seq))
    print('')
    
    print('='*20 + ' Unigram Language Model ' + '='*20)
    unigram_lm.fit(train_seqs)
    print('Train (Token) Perplexity: {}'.format(
        unigram_lm.perplexity_per_token(train_seqs))
    )
    print('Heldout (Token) Perplexity: {}'.format(
        unigram_lm.perplexity_per_token(heldout_seqs))
    )
    
    synth_seqs = unigram_lm.generate(num_examples_to_generate)
    print('==== Generated Examples ====')
    for seq in synth_seqs:
        print(' -> ' + ' '.join(seq))
    print('')
    
    #### TODO: Need to implement n-gram language model ####
    for n in [2, 3]:
        print('='*20 + ' {}-gram Language Model '.format(n) + '='*20)
        ngram_lm = NgramLM(n=n)

        print('Need to implement Ngram LM!\n')
        
        #ngram_lm.fit(train_seqs)
        #print('Train (Token) Perplexity: {}'.format(
        #    ngram_lm.perplexity_per_token(train_seqs))
        #)
        #print('Heldout (Token) Perplexity: {}'.format(
        #    ngram_lm.perplexity_per_token(heldout_seqs))
        #)
        #
        #synth_seqs = ngram_lm.generate(num_examples_to_generate)
        #print('==== Generated Examples ====')
        #for seq in synth_seqs:
        #    print(' -> ' + ' '.join(seq))
        #print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['sonnets', 'bible'],
                        help='which dataset to load: {"sonnets", "bible"}')
    parser.add_argument('--num_to_generate', type=int, default=1,
                        help='number of examples to generate')
    
    args = parser.parse_args()
    
    main(args.data, args.num_to_generate)
