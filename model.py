#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


# In[6]:


df =pd.read_csv('IMDB Dataset/IMDB Dataset.csv')
df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")


# In[10]:


for i in range(5):
    print("Review: ", [i])
    print(df['review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")


# In[11]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count


# In[12]:


df['word count'] = df['review'].apply(no_of_words)


# In[13]:


df.head()


# In[14]:


fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()


# In[15]:


fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['review'].str.len(), label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['review'].str.len(), label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()


# In[16]:


df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 0, inplace=True)


# In[17]:


df.head()


# In[18]:


def dataprocessing(text):
    text = text.lower()
    text = re.sub('<br />', '', text) 
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]','', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[19]:


df.review = df['review'].apply(dataprocessing)


# In[20]:


duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)


# In[21]:


df = df.drop_duplicates('review')


# In[22]:



def stemming(data):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in data]
    return data


# In[23]:


df.review = df['review'].apply(lambda x: stemming(x))


# In[24]:


df['word count'] = df['review'].apply(no_of_words)
df.head()


# In[25]:


pos_reviews = df[df.sentiment ==1]
pos_reviews.head()


# In[26]:


text = ' '.join([word for word in pos_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive reviews', fontsize = 19)
plt.show()


# In[27]:


from collections import Counter
count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)


# In[28]:


pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ['word', 'count']
pos_words.head()


# In[29]:


px.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color='word')


# In[30]:


neg_reviews = df[df.sentiment == 0]
neg_reviews.head()


# In[31]:


text = ' '.join([word for word in neg_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative reviews', fontsize = 19)
plt.show()


# In[32]:


count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] += 1
count.most_common(15)


# In[33]:


neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ['word', 'count']
neg_words.head()


# In[34]:


px.bar(neg_words, x='count', y='word', title='Common words in negative reviews', color='word')


# In[35]:


X = df['review']
Y = df['sentiment']


# In[36]:


df.head()


# In[37]:


vect = TfidfVectorizer()
X = vect.fit_transform(df['review'])


# In[38]:


X


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[ ]:





# In[40]:


print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# In[41]:


x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:500]
y_test = y_test[:500]


# In[42]:


print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# ### Building different model to analysis which model will best fit for dataset

# #### KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood.

# In[43]:


clf = KNeighborsRegressor(n_neighbors=19)
clf.fit(x_train,y_train)


# In[44]:


kn_reg = clf.score(x_test,y_test)


# #### This article concerns one of the supervised ML classification algorithm-KNN(K Nearest Neighbors) algorithm. It is one of the simplest and widely used classification algorithms in which a new data point is classified based on similarity in the specific group of neighboring data points. This gives a competitive result

# In[45]:


clf = KNeighborsClassifier(n_neighbors=19)
clf.fit(x_train,y_train)


# In[46]:


kn_clf = clf.score(x_test,y_test)


# #### DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset. In case that there are multiple classes with the same and highest probability, the classifier will predict the class with the lowest index amongst those classes.

# In[47]:


dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(x_train, y_train)


# In[48]:


dt_clf = dtree.score(x_test,y_test)


# In[51]:


print(dt_clf)


# #### The Random forest classifier creates a set of decision trees from a randomly selected subset of the training set. It is basically a set of decision trees (DT) from a randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction

# In[49]:


rf = RandomForestClassifier(n_estimators=100)
rf_clf = rf.fit(x_train, y_train)


# In[50]:


rf_clf = rf.score(x_test,y_test)
print(rf_clf)


# In[86]:


pickle.dump(rf, open('model.pkl','wb'))
pickle.dump(vect, open('vectorizer.pkl','wb'))


# In[93]:


# model = pickle.load(open('model.pkl','rb'))
# data_processing = pickle.load(open('data_processing.pkl','rb'))
# stemming = pickle.load(open('stemming.pkl','rb'))
# vectorizer = pickle.load(open('vectorizer.pkl','rb'))


# In[94]:


# test1 = "The story of first part was different and I think that's why many people liked it but for me it was only OK...as hero was only running here and there and getting frustrated and in the end we knew it was very simple case made complicated by a police officer only so i gave only 3* But 2nd part...I think it was fantastic, though the story had vibes that we have seen such stories earlier but still I find it quite amusing. I mean story is somewhat predictable but I don't think many will find the culprit before climax. Police checks the culprit in the very beginning only and we see him throughout movie so I love such concepts where killer is in front of us but we don't know that he is the killer.{Spoiler --> When Adivi sesh calls sanjana after seing her pic with cut on her throat,she says she is with Kumar and that directed us to believe that if he had not called her killer would have definitely killed her.So we can predict here that he is the killer at that point (just like me ;)}I also like the ending with NANI's entry [shadi hai ya godh bharai?😂] indicating he will be the protagonist in next part.Really excited for that🤩🤩🤩"
# test1 = data_processing(test1)
# test1 = stemming(test1)
# test1 = vectorizer.transform([test1])
# print(model.predict(test1))


# In[ ]:




