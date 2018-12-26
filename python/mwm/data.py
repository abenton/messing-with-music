'''
Methods to read & write data
'''

import os
import re
from urllib.request import urlopen

from mwm.paths import SONNETS_WEB, SONNETS_LOCAL
from mwm.paths import BIBLE_WEB, BIBLE_LOCAL


def prep_gutenberg_file(url, out_path, lowercase=True, force_overwrite=False):
    '''
    Downloads text file from project Gutenberg and tokenizes
    each line separately (strings of alphanumeric and
    non-alphanumeric characters, all punctuation kept).
    '''
    
    if (not os.path.exists(out_path)) or force_overwrite:
        response = urlopen(url)
        data = response.read()
        txt = data.decode('utf8')
        
        lns = txt.split('\n')
        
        lns_cleaned = []
        
        meat_begun = False
        for ln_idx, ln in enumerate(lns):
            if ln.strip().startswith('*** START OF THIS PROJECT GUTENBERG'):
                meat_begun = True
                continue
            
            if ln.strip().startswith('End of Project Gutenberg\'s'):
                break
            
            if meat_begun and ln.strip():
                lns_cleaned.append(ln.lower() if lowercase else ln)
        
        tokenized_lines = [[token.strip() for token in re.findall('\w+|\W+',
                                                                  ln)
                            if token.strip()]
                           for ln in lns_cleaned]
        
        with open(out_path, 'wt') as out_file:
            for tokens in tokenized_lines:
                out_file.write(' '.join(tokens) + '\n')
        

def load_sonnets():
    '''
    Load Shakespeare sonnets
    '''

    prep_gutenberg_file(SONNETS_WEB, SONNETS_LOCAL,
                        lowercase=True,
                        force_overwrite=False)
    
    with open(SONNETS_LOCAL, 'rt') as f:
        tokens = [[t for t in ln.strip().split()] for ln in f]
        return tokens


def load_bible():
    '''
    Load King James bible
    '''
    
    prep_gutenberg_file(BIBLE_WEB, BIBLE_LOCAL,
                        lowercase=True,
                        force_overwrite=False)
    
    with open(BIBLE_LOCAL, 'rt') as f:
        tokens = [[t for t in ln.strip().split()] for ln in f]
        return tokens
    
