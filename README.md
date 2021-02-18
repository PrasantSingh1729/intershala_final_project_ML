```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
df = pd.read_csv("churn_prediction.csv")
df.shape
```




    (28382, 21)



### Missing value and outlier treatment


```python
df=df[df['dependents']<7]
#plt.scatter(data['age'],data['dependents'])
df['dependents'].fillna(0,inplace=True);
df['gender'].fillna('Male',inplace=True);
df['occupation'].fillna('self_employed',inplace=True); 
df['days_since_last_transaction'].fillna(70,inplace=True); 
df=df.dropna()
# df = df.drop("customer_id",axis=1)
# df = df.drop("branch_code",axis=1)
```

### Variable Trasformation


```python
import math
# df['vintage']=np.sqrt(df['vintage'])
# df = df[df['dependents']<6]
# df['dependents'].value_counts()
```

### Model making


```python
df = pd.get_dummies(df)
```


```python
x = df.drop(['churn'], axis=1)
y = df['churn']
x.shape, y.shape
```




    ((25175, 25), (25175,))




```python
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56,stratify=y)
scaler = MinMaxScaler()
cols = train_x.columns
# cols
train_x_scaled = scaler.fit_transform(train_x)
train_x = pd.DataFrame(train_x_scaled, columns=cols)
test_x_scaled = scaler.transform(test_x)
test_x = pd.DataFrame(test_x_scaled, columns=cols)
# train_x
```

### Random forest


```python
clf = RandomForestClassifier(random_state=50, max_depth=6, n_estimators=76)
clf.fit(train_x,train_y)
temp=pd.DataFrame(clf.predict(train_x))
trainscore = f1_score(temp,train_y)
pred1=clf.predict(test_x)
testscore = f1_score(pred1,test_y)
print(trainscore,testscore)
print(clf.score(test_x,test_y))
```

    0.4876182806523051 0.49112426035502954
    0.8633619319987289
    

### Decision Tree


```python
clf = DecisionTreeClassifier(random_state=96,max_depth=4,min_samples_leaf=240)
clf.fit(train_x,train_y)
clf.score(train_x, train_y),clf.score(test_x, test_y)
temp=pd.DataFrame(clf.predict(train_x))
trainscore = f1_score(temp,train_y)
pred2=clf.predict(test_x)
testscore = f1_score(pred2,test_y)
print(trainscore,testscore)
print(clf.score(test_x,test_y))
```

    0.521919944550338 0.5360928823826351
    0.8539879250079441
    

### Logistic Regression


```python
logreg = LogReg(solver='saga')
logreg.fit(train_x, train_y)
temp=pd.DataFrame(logreg.predict(train_x))
trainscore = f1_score(temp,train_y)
pred3=logreg.predict(test_x)
temp=pd.DataFrame(logreg.predict(test_x))
testscore = f1_score(temp,test_y)
print(trainscore,testscore)
print(logreg.score(test_x,test_y))
```

    0.002320185614849188 0.003481288076588338
    0.8180807117890054
    

### K_NN


```python
k_nn=KNN(n_neighbors=5)
k_nn.fit(train_x,train_y)
temp=pd.DataFrame(k_nn.predict(train_x))
trainscore = f1_score(temp,train_y)
pred4=k_nn.predict(test_x)
temp=pd.DataFrame(k_nn.predict(test_x))
testscore = f1_score(temp,test_y)
print(trainscore,testscore)
print(k_nn.score(test_x,test_y))
```

    0.2701465201465202 0.08531468531468532
    0.792183031458532
    

## Ensemble


```python
from statistics import mode
final_pred = np.array([])
for i in range(0,len(test_x)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i],pred4[i]]))
```


```python
from sklearn.metrics import accuracy_score
print("Accuracy_score=",accuracy_score(test_y, final_pred))
print("F1_score=",f1_score(final_pred,test_y))
```

    Accuracy_score= 0.8603431839847474
    F1_score= 0.46823956442831216
    


```python

```
