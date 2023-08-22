import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import re
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
import pickle

#load the data
train = pd.read_csv("train.csv")


#data preprocessing

train.shape

train.isnull().sum()

train = train.dropna()

train.shape

#class imbalance in noticed here

train['sentiment'].value_counts()

#converting labels to numerical value (positive - 1, negative - 0)

label = LabelEncoder()
train['sentiment'] = label.fit_transform(train['sentiment'])

sns.countplot(x=train['sentiment'])

#text cleaning

def text_cleaner(text):
  text = text.lower()
  text = re.sub("[^a-zA-Z]", " ", text)
  return text

train['reviewText'] = pd.DataFrame(train.reviewText.apply(lambda x: text_cleaner(str(x))))

train.head()

#converting text data to numerical data

vectorizer = TfidfVectorizer(max_features = 5000)
rt = vectorizer.fit_transform(train['reviewText']).toarray()

Xtrain = pd.DataFrame(rt, columns = vectorizer.get_feature_names_out())
Ytrain = train['sentiment']
Xtrain.head()

#class balancing using undersampling

# undersampler =RandomUnderSampler()
# Xtrain_res , Ytrain_res = undersampler.fit_resample(Xtrain, Ytrain)

# Xtrain_res.shape

# Ytrain_res.shape

# sns.countplot(x=Ytrain_res)

#model training

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

def train_classifiers(estimator, Xtrain, Ytrain, cv, name):
  estimator.fit(Xtrain, Ytrain)
  cv_train_score = cross_val_score(estimator, Xtrain, Ytrain,
                                   cv=cv, scoring='f1_macro')
  print(f"On an average, {name} model has f1 score of "
        f"{cv_train_score.mean():.3f} +/- {cv_train_score.std():.3f} on the training set.")

# Logisitc Regression

# base_estimator = LogisticRegression(max_iter=1000)
# param_grid = {'C' : [100, 10, 1.0, 0.1, 0.01]}

# logreg_cv = GridSearchCV(base_estimator, param_grid,scoring='f1_micro', cv = 5)
# logreg_cv.fit(Xtrain,Ytrain)
# print("Tuned  Parameters: {}".format(logreg_cv.best_params_))
# print("Best score is {}".format(logreg_cv.best_score_))


logreg_cv= LogisticRegression(max_iter = 10000)
logreg_cv = logreg_cv.fit(Xtrain,Ytrain)

#saving model
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
pickle.dump(logreg_cv, open('model.pkl','wb'))

#loading model

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


#testing
def test_model(sentence): 
        sentence = [sentence]  
        sen = cv.transform(sentence).toarray()
        prediction = model.predict(sen)
        if prediction == 1:
            return 'Positive Review'
        elif prediction == 0:
            return 'Negative Review'

