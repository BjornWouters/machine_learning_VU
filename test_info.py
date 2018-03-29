from collections import Counter
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize, RegexpTokenizer
import csv

def count_words(file, stopwords):
    with open('text_output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'most_common_word', 'most_common_frequency', 'amount_of_words'])
        for line in word_file:
            line2 = line.split('||')
            ID = line2[0]
            tokenizer = RegexpTokenizer(r'\w+')
            tekst = tokenizer.tokenize(line2[len(line2) - 1])

            tekst2 = []
            for word in tekst:
                if word not in stopwords:
                    tekst2.append(word.lower())

            words = Counter()
            words.update(tekst2)

            mostcommonword = words.most_common(1)
            for word, frequency in mostcommonword:
                writer.writerow([str(ID), str(word), str(frequency), str(len(line.split()))])


stopwords = get_stop_words('en')
extra_stopwords = ["et", "al", "patient", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "cells", "cancer",
                   "patients"]
for extra in extra_stopwords:
    stopwords.append(extra)
word_file = open('dataset/stage2_test_text.csv', encoding="utf8")

count_words(word_file, stopwords)

