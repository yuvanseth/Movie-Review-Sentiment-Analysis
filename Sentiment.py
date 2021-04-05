import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
'''from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier'''
from sklearn.metrics import accuracy_score
import string


def clean(features):
    Features = []
    for review in features:
        words = nltk.tokenize.word_tokenize(review)

        words = [word.lower() for word in words]

        # remove punctuations 
        transform = str.maketrans('', '', string.punctuation)
        words = [word.translate(transform) for word in words]

        # remove non-alphabetical words
        words = [word for word in words if word.isalpha()]

        # remove uneccessary and overused prepositions like a, and, the
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]

        # combine list of words back to a review
        review = TreebankWordDetokenizer().detokenize(words)

        # convert all words to their root word according to meaning
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word, pos='v') for word in words]
        words = [lemmatizer.lemmatize(word, pos='a') for word in words]

        # remove line break tags
        review = review.replace('br', '')

        Features.append(review)
    
    return Features

# read file into a variable
ds = pd.read_csv('/Users/yuvanseth/Documents/VS/NLP_OCR/imdb_small.csv') 
# convert labels to numerical form
ds.loc[(ds.sentiment == 'positive'),'sentiment'] = 1
ds.loc[(ds.sentiment == 'negative'),'sentiment'] = 0
feat = ds.review
label = ds.sentiment
label = label.astype('int')

feat = clean(feat)

# to convert sentences to vectors using bag-of-words approach
cvect = CountVectorizer()
feat = cvect.fit_transform(feat)

# split the dataset into testing and training portions, apply the model, train, then calculate accuracy
feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size = 0.05)
model = LogisticRegression(max_iter=5002)
model.fit(feat_train, label_train)
predictions = model.predict(feat_test)

print(accuracy_score(label_test, predictions))

# Following are other classifier models in ascending order of their average score on this dataset
# DecisionTreeClassifier < svm.SVC < svm.LinearSVC < MultinomialNB < LogisticRegression < RandomForestClassifier
#        (0.65)             (0.82)      (0.82)          (0.86)            (0.87)                 (0.88)



