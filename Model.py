import pandas as pd

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#importing Data 


df= pd.read_csv("gooddata.csv")

#Sorting data and making sure i get dependet y value. 
dataset = df.values
x =dataset[:,:-2]
y =dataset[:,-1]




# Setting the Training and Testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0,)

# Scaler setup
sc = StandardScaler() 

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)

Reg_log = LogisticRegression()

Reg_log.fit(x_train,y_train)



y_predict = Reg_log.predict(x_test)

#Saving the model into a file.
dump(Reg_log  , filename="modelAI.joblib")



