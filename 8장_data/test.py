import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk     # nltk.download() 를 하기위해, import 함.

all_files = glob.glob('myCabinetExcelData*.xls')

all_files #출력하여 내용 확인