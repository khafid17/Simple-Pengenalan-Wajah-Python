import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
py.offline.init_notebook_mode(connected=True)
matplotlib

#loading the data
iris = load_iris()
In [3]

#setting up our x and y variables correspondingly
x=iris.data
y=iris.target

#concatinating the x and y np arrays into a single np array so that can be 
#converted to a dataframe later on
data=np.c_[x,y]

#making a header list for corresponding column indices in DF
cols=['sepal_length','sepal_width','petal_length','petal_width']
header=cols+['species']
#converting into a dataframe for visualisation purposes
iris_df=pd.DataFrame(data=data,columns=header)

iris_df.head()
#upadating values 0,1,2 in species column with real names
iris_df.species.replace(0.0,'iris-sesota',inplace=True)
iris_df.species.replace(1.0,'iris-versicolor',inplace=True)
iris_df.species.replace(2.0,'iris-virginica',inplace=True)

iris_df.shape

(150, 5)

'Some visualisations to understand the data better'

#import graph_obj as go
import plotly.graph_objs as go

#split the datasets according to the respective species so that
#easier while plotting scatter plots
df1=iris_df.iloc[:50,:]
df2=iris_df.iloc[50:100,:]
df3=iris_df.iloc[100:150,:]
#creating traces
trace1 = go.Scatter(
    #x=x-axis
    x=df1.sepal_length,
    #y=y-axis
    y=df1.sepal_width,
    #mode defines the type of plot eg-lines,markers,line+markers
    mode='markers',
    #name pf the plots
    name='iris-setosa',
    #markers->color and alpha of the respective trace
    marker=dict(color = 'rgba(255, 128, 2, 0.8)'),
    #the hover text
    text=df1.species)
trace2 = go.Scatter(
    x=df2.sepal_length,
    y=df2.sepal_width,
    mode='markers',
    name='iris-versicolor',
    marker=dict(color = 'rgba(0, 255, 200, 0.8)'),
    text=df2.species)
trace3 = go.Scatter(
    x=df3.sepal_length,
    y=df3.sepal_width,
    mode='markers',
    name='iris-virginica',
    marker=dict(color = 'rgba(255, 128, 255, 0.8)'),
    text=df3.species)
#a list of all the traces
data_list=[trace1,trace2,trace3]
#it is a dictionary containing info about title,axis etc
layout=dict(title='Sepal length and Sepal Width of Species',
               xaxis=dict(title='Sepal Length',ticklen=5,zeroline=False),
               yaxis=dict(title='Sepal Width',ticklen=5,zeroline=False)
            )
#fig object includes data and layout
fig=dict(data=data_list,layout=layout)
#plotting the fig
py.offline.iplot(fig)

#import graph_obj as go
import plotly.graph_objs as go

#split the datasets according to the respective species so that
#easier while plotting scatter plots
df1=iris_df.iloc[:50,:]
df2=iris_df.iloc[50:100,:]
df3=iris_df.iloc[100:150,:]
#creating traces
trace1 = go.Scatter(
    #x=x-axis
    x=df1.petal_length,
    #y=y-axis
    y=df1.petal_width,
    #mode defines the type of plot eg-lines,markers,line+markers
    mode='markers',
    #name pf the plots
    name='iris-setosa',
    #markers->color and alpha of the respective trace
    marker=dict(color = 'rgba(255, 128, 2, 0.8)'),
    #the hover text
    text=df1.species)
trace2 = go.Scatter(
    x=df2.petal_length,
    y=df2.petal_width,
    mode='markers',
    name='iris-versicolor',
    marker=dict(color = 'rgba(0, 255, 200, 0.8)'),
    text=df2.species)
trace3 = go.Scatter(
    x=df3.petal_length,
    y=df3.petal_width,
    mode='markers',
    name='iris-virginica',
    marker=dict(color = 'rgba(255, 128, 255, 0.8)'),
    text=df3.species)
#a list of all the traces
data_list=[trace1,trace2,trace3]
#it is a dictionary containing info about title,axis etc
layout=dict(title='Petal length and Petal Width of Species',
               xaxis=dict(title='Petal Length',ticklen=5,zeroline=False),
               yaxis=dict(title='Petal Width',ticklen=5,zeroline=False)
            )
#fig object includes data and layout
fig=dict(data=data_list,layout=layout)
#plotting the fig
py.offline.iplot(fig)

#the correlation matrix heatmap for analysing the correlation among features
corr_martix=iris_df[cols].corr()
sns.heatmap(corr_martix,cbar=True,annot=True,fmt='.1f',cmap='coolwarm');
#we can see petal length and petal width ad correlated very highly and same can be said for sepal lenght and sepal width

class Question:
    #initialise column and value variables->
    #eg->if ques is ->is sepal_length>=1cm then
    #sepal_length==col and 1cm=value
    def __init__(self,column,value):
        self.column=column
        self.value=value
    #it matches wheter the given data is in accordace with the value set or not
    #returns true and false accordingly
    def match(self,data):
        value=data[self.column]
        return value>=self.value
    # This is just a helper method to print
    # the question in a readable format.
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


    'demo of question class'
#forming a question
Question(0,5)
## it takes column as 0 and value as 5 
q=Question(0,5)
#now it checks wheter the values on 0th column of the 4th datapoint is >= 5 or not
#and returns true or false accordingly
q.match(x[3])


false

#count the unique values of labels and store them in a dictionary
def count_values(rows):
    #will return a dictionary with species values as key and frequency as values
    count={}
    #takes whole dataset in as argument
    for row in  rows:
        #traverse on each datapoint
        label=row[-1]
        #labels are in the last column
        #if label is not even once come initialise it
        if label not in count:
            count[label]=0
        #increase the count of present label by 1
        count[label]+=1
    return count


'''demo count function'''
count_values(data)
#hinglish comment
#haar row main jayega -> last element ko label se initialise karega->


{0.0: 50, 1.0: 50, 2.0: 50}


#spliting the data based on the respective ques.
def partition(rows,question):
    #intialise two seprate lists 
    true_row,false_row=[],[]
    for row in rows:
        #traverse on each datapoint
        #match the given datapoint with the respective question
        if question.match(row):
            #if question.match returns true aka value is satisfied
            #append the given row in true row list
            true_row.append(row)
        else:
            false_row.append(row)
    return true_row,false_row


#demo of partition function
#our question is ->
print(Question(0,5))
#t_r represents true_rows and f_r false_rows
t_r,f_r=partition(data,Question(0,5))
#thus t_r will only contain sepal legnth values > 5cm
t_r
