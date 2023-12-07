#removing the columns which will little to no effect on the dataset
# in the data from what i have noticed is that sex se survival rate change krra h bhut jor se
#passenger class se survival rate change krra h bhut jor se
#cabin and number of sibling have almost null effect
# minr effect of place from where they climbed because jo port s h vahan se number of people jo chadhe h vo zyada h..
#those with expensive tickets had a higher probability of surving
#while checking the cabin i found that cabin b d and e had higher surving rate but i am not sure due to excessive null valuses present in the particular data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#_test stands for test data

df = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
df_check = pd.read_csv('./gender_submission.csv')
column_names = list(df.columns)
# print(column_names)
df.pop(column_names[3])
df_test.pop(column_names[3])
df.pop(column_names[7])
df_test.pop(column_names[7])
df.pop(column_names[8])
df_test.pop(column_names[8])
df.pop(column_names[10])
df_test.pop(column_names[10])
df = df.drop(df[(df.PassengerId == 62) &
                         (df.PassengerId == 830)].index)

x = df.iloc[:, 2:].values
x_test = df_test.iloc[:, 1:].values
y = df.iloc[:, 1].values
y_test = df_check.iloc[:, 1].values
#taking care of the missing values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(x[:, 2:5])
x[:, 2:5] = impute.transform(x[:, 2:5])
x_test[:, 2:5] = impute.transform(x_test[:, 2:5])

# label encoding the genders as 1 and 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])
x_test[:, 1] = le.transform(x_test[:, 1])


# Scaling the data so it comes in the same range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x[:, 2:5] = sc.fit_transform(x[:, 2:5])
x_test[:, 2:5] = sc.transform(x_test[:, 2:5])

#Onehot encoding the ports to remove any biases caused due to assumed relation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x_test = np.array(ct.transform(x_test))


#training the actual model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(x, y)
y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x,y = y, cv=10)
print(f'Accuracy: {accuracies.mean()*100}%')
print(f'Standard Devaition: {accuracies.std():.2}')