#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[1]:


# Packages
import pandas as pd
import numpy as np

imdb_data = pd.read_table('C:/Users/Ntando/Downloads/sentiment labelled sentences/imdb_labelled.txt', names= ['sentence', 'sentiment'])


# In[2]:


# a quick view of the first 5 rows
imdb_data.head()


# In[3]:


# we get the dimension of the dataset
imdb_data.shape


# In[4]:


# function to get word length per row
def word_count(string):
    # get the word split
    word = string.split()

    # get the length
    length = len(word)
    # return length of word
    return length

imdb_data['word_len'] = imdb_data['sentence'].apply(word_count)
imdb_data.head() 


# In[5]:


# Number of characters feature
imdb_data['num_characters'] = imdb_data['sentence'].apply(len)
imdb_data.head() 


# In[6]:


# function to get the average word length
def avg_word_length(string):
    # get word lengths
    words = string.split()
    words_lengths = [len(word) for word in words]

    # get the average
    avg_word_length = sum(words_lengths)/len(words)
    return avg_word_length

imdb_data['avg_characters'] = imdb_data['sentence'].apply(avg_word_length)
imdb_data.head() 


# In[7]:


# function to get the number of sentences
from nltk.tokenize import sent_tokenize, word_tokenize

def num_sentences(string):
    # make sentences
    sentences = sent_tokenize(string)
    length = len(sentences)
    return length


imdb_data['num_sentence'] = imdb_data['sentence'].apply(num_sentences)
imdb_data.head()


# In[8]:


import re

# function to get number of words stating with capital letters
def capital_letters(string):
    # the capital words
    capital_word = r"[A-Z]"
    capital_words = re.findall(capital_word, string)
    number_words = len(capital_words)
    return number_words

imdb_data['num_capital_letters'] = imdb_data['sentence'].apply(capital_letters)
imdb_data.head()


# In[9]:


# function of getting number of digits
def digits(string):
    digits_sent = r"[0-9]"
    digits_ = re.findall(digits_sent, string)
    digits = len(digits_)
    return digits

imdb_data['num_digits'] = imdb_data['sentence'].apply(digits)
imdb_data.head()


# In[10]:


# Which sentiments have more than 10 digits within it?
imdb_data[imdb_data['num_digits'] > 10].head(5)


# In[11]:


# We examine the 136th row
imdb_data.iloc[136, 0]

