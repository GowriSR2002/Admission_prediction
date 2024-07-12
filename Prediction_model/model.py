#import library
import pandas as pd

# Read dataset
df=pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Admission%20Chance.csv')

# define y and X
y=df['Chance of Admit ']
X=df[['GRE Score', 'TOEFL Score', 'University Rating', ' SOP',
       'LOR ', 'CGPA', 'Research']]

#split to train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2529)

#Model selection
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#training
model.fit(X_train,y_train)
#Prediction
y_pred=model.predict(X_test)

#Evaluation
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)

import pickle
name='model.sav'
pickle.dump(model,open(name,'wb'))

load_model=pickle.load(open('model.sav','rb'))