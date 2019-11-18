import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# %mat-plot-lib inline

# Stage : Data Processing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
pd.set_option('display.max_columns', None)

# show only those tweets that are labelled as offensive
variable = train[train['label'] == 1]

# Stage : Data Cleaning
# remove twitter handles, punctuation, numbers and special char, smaller words
# To save time, we'll combine our train and test data frames and do cleaning on both

# combine csv files
combi = train.append(test, ignore_index=True)


# Removing Twitter handles
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for l in r:
        input_txt = re.sub(l, '', input_txt)

    return input_txt


combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], '@[\w]*')
# Removing special characters, numbers, punctuation
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace('[^a-zA-Z#]', " ")

# Removing short words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Tokenizing
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

# Stemming
from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(k) for k in x])

# check stemmed tweets
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet

# Data visualization
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Words in positive tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Words in negative tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=500, height=800, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


def hashtag_extract(x):
    hashtags = []
    for j in x:
        ht = re.findall(r'#(\w+)', j)
        hashtags.append(ht)

    return hashtags


HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

HT_regular = sum(HT_regular, [])

HT_negative = sum(HT_negative, [])

# Plot most common hash_tags
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x='Hashtag', y='Count')
ax.set(ylabel='Count')
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

e = e.nlargest(columns='Count', n=10)
ax = sns.barplot(data=d, x='Hashtag', y='Count')
ax.set(ylabel='Count')
plt.show()

# Extracting Features

# Extract Bag-of-Words
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

# Extract with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

# Model Building

# Model with the bag-of-words data-frame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print('f1_score: ', f1_score(yvalid, prediction_int))

# Model with TF-IDF data-frame
train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print('f1_score: ', f1_score(yvalid, prediction_int))
