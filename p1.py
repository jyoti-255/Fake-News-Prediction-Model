#import lib
import pandas as pd
import pandas as pd
import re
from nltk.corpus import stopwords   #the,for,of,in,with
from nltk.stem.porter import PorterStemmer   #loved,loving==love
from sklearn.feature_extraction.text import TfidfVectorizer   #loved=[0 0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load the data
data=pd.read_csv('train.csv')
print(data)

print(data.shape)

print(data.isnull().sum())
data=data.fillna(' ')

print(data.isnull().sum())

data['content']=data['author']+" "+data['title']
print(data)

#preprocessing

if 'content' in data.columns:
    if 20796 < len(data):
        print(data['content'].iloc[20796])
    else:
        print("Index 20796 is out of bounds.")
else:
    print("'text' column does not exist.")


#stemming
ps=PorterStemmer()

def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]'," ",content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content="  ".join(stemmed_content)
    return stemmed_content


data['content']=data['content'].apply(stemming)
print(data['content'].values)


#separating the dataset
x=data['content'].values
y=data['label'].values

print(x)
print(y)

vector=TfidfVectorizer()
vector.fit(x)
x=vector.transform(x)
print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print(x_train.shape)

#model
model=LogisticRegression()
model.fit(x_train,y_train)

train_y_pred=model.predict(x_train)
print("train accuracy:",accuracy_score(train_y_pred,y_train))

test_y_pred=model.predict(x_train)
print("test accuracy:",accuracy_score(train_y_pred,y_train))



#prediction system

input_data=x_test[100]
prediction=model.predict(input_data)
if prediction[0]==1:
    print('Fake news')
else:
    print('Real news')

    