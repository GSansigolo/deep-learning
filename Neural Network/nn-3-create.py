'''
Created on Sep 08, 2018
Author: @G_Sansigolo
'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as numpy
import random
import pickle
from collection import Couter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


