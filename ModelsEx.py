
import pandas
import matplotlib.pyplot as plt

from sklearn import model_selection #breaks our data in x and y




url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=cols)

#df = pandas.read_csv(file_path)
#df = pandas.read_csv(file_path,name=['col1','col2'])

#print(dataset)

print(dataset.shape)
print(dataset.head()) #default 5 rows 
print(dataset.tail()) #default 5 rows 



#stats
print(dataset.describe())

#distribution
print(dataset.groupby('class').size())



#visualiation
dataset.plot(kind='box', subplots=True, layout=(1,4), sharex=False, sharey=False)
plt.show()




'''
features extraction :
    feature: is independent columns
    Example:
          Iris-setosa :
                      hieght,
                      color
                      width


x   is input (independent)
y   is response (output/dependent variable)

Marks:
    hours(x)  marks(y)
     3         40


x = [sepal-length  sepal-width  petal-length  petal-width]
y = [class]


Train(historical data), and test data(data to be validated)
--------------
split dataset in two part (train, test)
         20% test, 80 % train
         30%  - 70%
            



'''


array = dataset.values


X = array[:,0:4] #all rows and 0 to 3 columns (4 numeric columns)
Y = array[:,4]  #all rows and 4th column (class)

print(X)
print(Y)



#

x_train, x_validate, y_train, y_validate= model_selection.train_test_split(X, Y, test_size=.20, random_state=7)

print(x_train)

print(x_validate)
print(y_train)
print(y_validate)

print(len(x_train))
print(len(x_validate))
      



