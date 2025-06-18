
# import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#load dataset
dataset = pd.read_csv(r"C:\Users\rutik\Downloads\Restaurant_Reviews - Restaurant_Reviews.tsv",delimiter='\t',quoting=3)


import re 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#blank list created
corpus=[]

#take to proper formating
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review =review.split()
    ps =PorterStemmer()
    review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#creating Bag of word model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
'''

#creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x =cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
'''

#split dadta into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
'''
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier()
classifier.fit(x_train,y_train)
'''


'''
#SVC Classifier
from sklearn.svm import SVC
classifier =SVC()
classifier.fit(x_train,y_train)
'''

#Logistic regression
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(x_train,y_train)
 
 
'''
 #Knn classifier
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)
'''


'''
#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier()
classifier.fit(x_train,y_train)
'''

'''
#XGBClassifier
from sklearn.xgboost import XGBClassifier
classifier =XGBClassifier()
classifier.fit(x_train,y_train)
'''
'''
#BernoulliNB
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(x_train,y_train)
'''
'''
#GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train,y_train)
'''
'''
#LGBMClassifier
from sklearn.lightgbm import LGBMClassifier
classifier = LGBMClassifier()
classifier.fit(x_train,y_train)
'''
#predicted
y_pred = classifier.predict(x_test)

#confusion metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_test, y_pred)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)


#bias
bias =classifier.score(x_train,y_train)
bias


#variance
variance = classifier.score(x_test,y_test)
variance
    

