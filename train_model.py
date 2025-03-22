import pandas as pd
import io
from google.colab import files


#uploaded = files.upload()  
#uploaded.keys()
df = pd.read_csv(io.BytesIO(uploaded['music.csv']))

# Read the uploaded CSV file
# Show first 5 rows
df
X = df.drop(columns =['genre'])
Y = df['genre']
Y


from sklearn.tree import DecisionTreeClassifier #LIBRARY, MODULE , CLASS
model = DecisionTreeClassifier()
model.fit(X,Y)
predictions= model.predict([[21,1], [22, 0]])
predictions


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#80 percent for training
#20 percent for test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
score = accuracy_score(Y_test, predictions)

predictions
score




import joblib
#joblib.dump(model, 'music-recommender.joblib')
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21,1]])
predictions

from sklearn import tree
tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(Y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
