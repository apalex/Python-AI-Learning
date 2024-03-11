import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/

# NumPy Arrays
mylist = [1, 2, 3]
type(mylist) #prints out list

np.array(mylist)
my_matrix = [[1,2,3], [4,5,6], [7,8,9]]
np.array(my_matrix)

np.arange(0, 10) #Creates a NumPy array that begins at 0, and ends at 9 with a size of 10
np.arange(0, 10, 2) #Adds jump size of 2
np.zeros((2,5)) #Creates a 2x5 dimensional array with decimal zeros
np.ones((6,6)) #Creates a 6x6 dimensional array with decimal ones
np.linspace(0, 10, 10) #Start=0, End=10, EvenlySpacedNumbers=10
np.eye(5) #Creates an Identity Matrix
np.random.rand(1) #Returns a random number between 0 and 1
np.random.rand(5,5) #Creates a 5x5 matrix with random numbers
np.random.randn(2,3) #Returns a standard normal distribution with mean 0 and variance 1
np.random.randint(0, 101, 10) #Returns 10 random Integers between 0 and 101
np.random.randint(0, 101, (4,5)) #Returns 10 random Integers between 0 and 101 in a 4x5 array
np.random.seed(42) #Returns a set of numbers in .rand
np.random.rand(4) #Result of .seed
arr = np.arange(0,25)
arr.reshape(5,5) #Reshapes array into a 5x5 array
randarr = np.random.randint(0, 101, 10)
randarr.max() #Returns max Number in array
randarr.min() #Returns min Numberin array
randarr.argmax() #Returns index of max number
randarr.argmin() #Returns index of min number
randarr.dtype #Returns type of array
arr.shape #Returns shape of the array

################################################################################################

# Indexing and Selection
# If you do slice_of_arr = arr[:5]
# And you edit slice_of_arr then it also edits the original arr
# To avoid this, use the command slice_of_arr = arr.copy() to create a copy
 
# Operations
# new_arr = np.arange(0, 10)
# new_arr += 5 #Will addition every element inside the array by 5
# new_arr += new_arr
# arr2d = np.arange(0,25).reshape(5,5)
# arr2d.shape
# arr2d.sum(axis=0) #Additions all the columns
# arr2d.sum(axis=1) #Additions all the rows

################################################################################################

# Matplotlib
# x = np.arange(0, 10)
# y = 2*x
# plt.plot(x, y)
# plt.ylabel('Y-axis')
# plt.xlabel('X-axis')
# plt.title('Linear Graph')
# plt.xlim(0,10)
# plt.ylim(0,10)
# plt.show()
#help(plt.savefig)
#help(plt.figure)
#plt.savefig('linear.png')

# fig = plt.figure() #Creates a blank canvas 432x288 with 0 Axes
# plt.figure(figsize=(10, 10))
# axes = fig.add_axes([0,0,1,1])
# axes.plot(x,y)

# a = np.linspace(0, 10, 11)
# b = a**4
# c = np.arange(0,10)
# d = 2*c
# add = fig.add_axes([0,0,1,1])
# add.plot(c,d)

#fig, axes = plt.subplots(nrows=2,ncols=2)
#plt.tight_layout()
#axes[0][0].plot(x,y)
#fig.subplots_adjust(wpsace=1,hspace=0.5)
#fig.set_figwidth(10)

# x5 = np.linspace(0,11,10)
# fig5 = plt.figure()
# ax5 = fig5.add_axes([0,0,1,1])
# ax5.plot(x5,x5, label='X vs X')
# ax5.plot(x,x**3, label='X vs X',color='blue',lw=2,ls='-',marker='+',ms=5)
# ax5.legend(loc=(1.1,0.5))
# plt.show()

################################################################################################

#Exercises

#1
# m = np.linspace(0,10,11)
# c = 3* 10**8
# E = m* c**2

# #2
# plt.plot(m,E,color='red')
# plt.title('E=mc**2')
# plt.xlabel('Mass in g')
# plt.ylabel('Energy in J')
# plt.xlim(0,10)

# #3
# plt.yscale('log')
# plt.grid()
# plt.show()

################################################################################################

#Theory
#ML = Machine Learning
#DL = Deep Learning
#RL = Reinforcement Learning
#AI = Artificial Intelligence

# - Artificial Intelligence -
# Intelligence demonstrated by machine
# Turing Test
# Marcus Test
# Lovelace 2.0 Test
# General AI = Human Level (or better) intelligence in multiple domains
# Narrow AI = Human level intelligence in a specific domain (e.g: chatbot, image recognition, etc)

# - Machine Learning -
# Subdomain of AI
# 3 Types of ML
# 1. Supervised Learning (SL)
# - Uses supervised historical labeled data to predict future outcomes
# 2. Unsupervised Learning (UL)
# - Uses historical non-labeled data to discover patterns in data
# 3. Reinforcement Learning (RL)
# - Does not rely on historical data. Relies on agent, environment, observation, rewards to learn.

# - Deep Learning (Artificial Neural Networks)
# Related to ML's Deep Learning

# ML Pathway
# Real World -> Collect & Store Data -> Clean & Organize Data -> Exploratory Data Analysis ->
# ML Models (SL=Predict Outcome or UL=Discover Patterns in Data)

# Supervised ML Processgi
# Many algorithms have adjustable values
# Data -> X: Features, y: Label -> 1. Training Data Set | 2. Test Data Set
# 1. Training Data Set -> Fit/Train Model -> Adjust as Needed OR Evaluate Performance
# 2. Test Data Set -> Evaluate Performance
# Evaluate Performance -> Deploy Model as Service/Dashboard/App -> Data Product -> Real World

################################################################################################

#Pandas
myindex = ['Canada', 'Uganda', 'Australia']
mydata = ['CA', 'UG', 'AU']
myser = pd.Series(data=mydata, index=myindex)
print(myser)
print(myser[0])
print(myser['Uganda'])
print(myser.keys())

ages = {'Jean-Paul':5,'Francois':10,'Genevieve':50}
pd.Series(ages)

mycompanies = ['Microsoft', 'Google', 'Nvidia']
sales = [0.5, 2, 10]

mycompanies1 = ['Microsoft', 'Google', 'Nvidia']
sales1 = [2, 3, 20]

sales_q1 = pd.Series(index=mycompanies,data=sales)
sales_q2 = pd.Series(index=mycompanies1,data=sales1)
total_sales = sales_q1.add(sales_q2,fill_value=0)
print(total_sales)

#Dataframes in Pandas

# np.random.seed(20)
# random_data = np.random.randint(0, 101, (4,4))
# print(random_data)
# my_provinces = ['QC', 'ON', 'AB', 'VA']
# my_columns = ['Jan', 'Feb', 'Jun', 'Jul']
# df = pd.DataFrame(data=random_data,index=my_provinces, columns=my_columns)
# df.set_index('Jun')
# df.reset_index()
# drop = df.drop('Jul',axis=1)
# df.iloc[0:2]
# one_row = df.iloc[1]
# df = df.append(one_row)
# df.iloc[['QC', 'VA']]
# df.loc['Jan']

# print(drop)
# print(df)
# print(df.info())
# print(df.describe())
# print(df.describe().transpose())
# print(df.head(2))
# print(df.columns)
# print(df.index)
# print(df['Jan'])

#Files with Pandas
#read = pd.read_csv('C:\\Users\\username\\file_path_here.csv')

################################################################################################

#Scikit-learn

# Train | Test Split Procedure
# 0. Clean and adjust data as necessary X and y
# 1. Split Data in Train/Test for both X and y
# 2. Fit/Train Scaler on Training X Data
# 3. Scale X Test Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6

#model = Ridge(alpha=100)
#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#print(mean_squared_error(y_test, y_pred))
#model2 = Ridge(alpha=1)
#model2.fit(X_train,y_train)
#y_pred_two = model2.predict(X_test)

# df = pd.read_csv('Advertising.csv')
# df.head()
# X = df.drop('sales', axis=1)
# y = df['sales']
# X_train,X_other,y_train,y_other = train_test_split(X,y,test_size=0.3,random_state=101)
# X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5,random_state=101)
# print(len(df))
# print(len(X_train))
# print(len(X_eval))
# print(len(X_test))
# # Scaling X Test Data
# scaler = StandardScaler()
# scaler.fit(X_train)
# # Order to scale transform does not matter
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# X_eval = scaler.transform(X_eval)
# # Creating Model
# model_one = Ridge(alpha=100)
# model_one.fit(X_train, y_train)
# y_eval_pred = model_one.predict(X_eval)
# print(mean_squared_error(y_eval, y_eval_pred))

# model_two = Ridge(alpha=1)
# model_two.fit(X_train, y_train)
# new_pred_eval = model_two.predict(X_eval)
# print(mean_squared_error(y_eval, new_pred_eval))

# y_final_test_pred = model_two.predict(X_test)
# print(mean_squared_error(y_test, y_final_test_pred))

################################################################################################

#Artificial Neural Networks

#Non-deep neural network contain 1 hidden layer
#Deep Neural network contains 2+ hidden layers

#Composition of Neural Network
#Input Layer: First layer that directly accepts real data values
#Hidden Layer: Any layer between input and output layers
#Ouput Layer: The final estimate of the output

# Perceptron Model Formula:
# z = x*w+b
# w=weight (weight or strenght given to input/importance)
# b=bias (offset/treshold value)

#Activiation Functions:
#Binary Step
#Step Function
#Dynamic Function = Sigmoid Function
#Hyperbolic Tangent
#Rectified Linear Unit (ReLU)

#Organizing Multiple Classes
#One-hot encoding

#Softmax Function:
#Calculates the probabilities distribution of the event over K different events
#(Calculates the probabilities of each target class over all possible target classes)
#Sum of all probabilities will be equal to 1

#o(z)=a
#y=true value
#a=neuron's prediction
#z=activation function

#Cost Functions:
#Quadratic Cost Function: Calculate difference between real values against predicted values

#Cost Function consists of:
#C(W,B,Sr,Er)
#W=Neural Networks weight
#B=Neural Networks biases
#Sr=Input of single traning sample
#Er=Desired output of that training sample

#Cross Entropy Loss Function

#Backpropogation

################################################################################################

#Keras
df = pd.read_csv('fake_reg.csv')
print(df.head())
sns.pairplot(df)
X = df[['feature1','feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Dense(unit=4,activation='relu'),
    Dense(unit=2,activation='relu'),
    Dense(unit=1)
    ])
