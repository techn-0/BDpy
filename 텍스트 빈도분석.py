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
#필요한 패키지 임포트

all_files = glob.glob('c:/Users/gnlrj/Desktop/My_Python/8장_data/myCabinetExcelData*.xls')
#스파이더에서 실행시 매번 경로 설정이 필요해 절대경로로 변경하였음

all_files #출력하여 내용 확인

all_files_data = [] #저장할 리스트 
#--------------------------------------------------------
for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)

all_files_data[0] #출력하여 내용 확인
#--------------------------------------------------------
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

all_files_data_concat #출력하여 내용 확인
#--------------------------------------------------------
all_files_data_concat.to_csv('c:/Users/gnlrj/Desktop/My_Python/8장_data/riss_bigdata.csv', encoding='utf-8', index = False)

#--------------------------------------------------------
#데이터 전처리
# 제목 추출
all_title = all_files_data_concat['제목']

all_title #출력하여 내용 확인
#--------------------------------------------------------
stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()



words = []  

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))    
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)



    print(words)  #출력하여 내용 확인. 크기가 너무 커서 기다려야함 멈춤

    words2 = list(reduce(lambda x, y: x+y,words))
    print(words2)  #작업 내용 확인


    #데이터 탐색

    count = Counter(words2)

count   #출력하여 내용 확인

word_count = dict()

for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))



del word_count['big']
del word_count['data']

#히스토그램
plt.figure(figsize=(12,5))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='85')

plt.show()

#그래프 그리기
#연도별데이터
all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
summary_year  #출력하여 내용 확인



plt.figure(figsize=(12,5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)

plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])

plt.show()

#워드 클라우드
stopwords=set(STOPWORDS)
wc=WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()