```python
import pandas as pd
dt_train=pd.read_csv('diabetes.csv')
dt_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dt_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
print(dt_train.shape)
```

    (768, 9)
    


```python
print((dt_train['Glucose']==0).sum())
```

    5
    


```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot(x='Glucose',kind="bar",data=dt_train)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-5-40893a2454e1> in <module>
          1 import matplotlib.pyplot as plt
          2 import seaborn as sns
    ----> 3 plt.plot(x='Glucose',kind="bar",data=dt_train)
    

    ~\anaconda3\lib\site-packages\matplotlib\pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
       2794     return gca().plot(
       2795         *args, scalex=scalex, scaley=scaley, **({"data": data} if data
    -> 2796         is not None else {}), **kwargs)
       2797 
       2798 
    

    ~\anaconda3\lib\site-packages\matplotlib\axes\_axes.py in plot(self, scalex, scaley, data, *args, **kwargs)
       1663         """
       1664         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
    -> 1665         lines = [*self._get_lines(*args, data=data, **kwargs)]
       1666         for line in lines:
       1667             self.add_line(line)
    

    ~\anaconda3\lib\site-packages\matplotlib\axes\_base.py in __call__(self, *args, **kwargs)
        169             if pos_only in kwargs:
        170                 raise TypeError("{} got an unexpected keyword argument {!r}"
    --> 171                                 .format(self.command, pos_only))
        172 
        173         if not args:
    

    TypeError: plot got an unexpected keyword argument 'x'



![png](output_4_1.png)



```python
import seaborn as sns
sns.catplot(x='Glucose',col='Outcome',kind='count',data=dt_train)
```




    <seaborn.axisgrid.FacetGrid at 0x2e424d7948>




![png](output_5_1.png)

sns.countplot(x='Outcome',data=dt_train)

```python
sns.countplot(x='Glucose',data=dt_train.iloc[:15])
```


```python
print((dt_train['Glucose']==78).sum())
```


```python
dt_train['Glucose'][:10]
```


```python
import numpy as np
diabetes_data_copy = dt_train.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())
```


```python
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
```


```python
p = diabetes_data_copy.hist(figsize = (20,20))
```


```python
x=[1,1,1]
y=[4,5,6]
plt.plot(x,y)
plt.xlabel('indices')
plt.ylabel('values')
plt.title('matplot')
plt.grid()
plt.show()
```


```python
plt.plot(x,y,'ro')
plt.grid()
```


```python
import numpy as np
t=np.arange(0,5,2)
print(t)
```


```python
plt.plot(t,t**2,'b--',label="^2")
plt.plot(t,t**3,'rs',label="^3")
plt.grid()
plt.legend()
plt.show()
```


```python
t2=np.arange(2,10,2)
print(t2)
```


```python
plt.figure(1)
plt.subplot(211)
plt.grid()
plt.plot(t2,t2**2,'r^')
plt.subplot(212)
plt.plot(t2,t2**3,'gs')
plt.grid()
plt.show()
```


```python
plt.figure(1)
plt.subplot(411)
plt.grid()
plt.plot(t,t**2,'r--',label='^2')
plt.legend()
plt.subplot(412)
plt.grid()
plt.plot(t,t**3,'g--',label='^3')
plt.legend()
plt.subplot(421)
plt.grid()
plt.plot(t,t**4,'g^',label='^4')
plt.legend()
plt.subplot(422)
plt.grid()
plt.plot(t,t+2,'y^',label='+2')
plt.legend()
plt.show()

```


```python
sns.pairplot(dt_train,hue='Outcome')
```


```python
p=dt_train.hist(figsize=(20,20))
```


```python
print((diabetes_data_copy['BloodPressure']==0).value_counts())
```


```python
print((dt_train['BloodPressure']==0).value_counts())
```


```python
print(diabetes_data_copy['SkinThickness'].isnull().sum())
```


```python
dt_train["Outcome"].value_counts()
```


```python
sns.countplot(x="Outcome",data=dt_train)
```


```python
#scatterplot
dt_train.plot(kind="scatter",x="SkinThickness",y="Glucose")
plt.show()
```


```python

```


```python
sns.set_style("whitegrid");
sns.FacetGrid(dt_train,hue="Outcome",size=5) \
      .map(plt.scatter,"Glucose","SkinThickness") \
       .add_legend();
plt.show()

```


```python
import numpy as np
print(np.mean(dt_train['Glucose']))
```


```python
print(np.std(dt_train['Glucose']))
```


```python
dt_train.describe().T
```


```python
print(np.percentile(dt_train["Glucose"],(25,50,75,100)))
```


```python
print(np.median(dt_train["Glucose"]))
```


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
```


```python
y=diabetes_data_copy.Outcome
```


```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42,stratify=y)

```


```python
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

score_list=[]
for each in range(1,11):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test,y_test))
    #print(each,score_list[each-1])

plt.plot(range(1,11),score_list)
plt.xlabel("k-values")
plt.ylabel("Accuracy")
plt.show()
```


```python
score_list.index(max(score_list))
```


```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```


```python
y_pred = knn.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)
```


```python

plt.figure(figsize=(10,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
```


```python
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred,rownames=['Tru'], colnames=['Predicte'], margins=True)
```


```python

```


```python

```
