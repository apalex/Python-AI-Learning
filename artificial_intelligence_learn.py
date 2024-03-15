import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os

#https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/

# NumPy Arrays
# mylist = [1, 2, 3]
# type(mylist) #prints out list

# np.array(mylist)
# my_matrix = [[1,2,3], [4,5,6], [7,8,9]]
# np.array(my_matrix)

# np.arange(0, 10) #Creates a NumPy array that begins at 0, and ends at 9 with a size of 10
# np.arange(0, 10, 2) #Adds jump size of 2
# np.zeros((2,5)) #Creates a 2x5 dimensional array with decimal zeros
# np.ones((6,6)) #Creates a 6x6 dimensional array with decimal ones
# np.linspace(0, 10, 10) #Start=0, End=10, EvenlySpacedNumbers=10
# np.eye(5) #Creates an Identity Matrix
# np.random.rand(1) #Returns a random number between 0 and 1
# np.random.rand(5,5) #Creates a 5x5 matrix with random numbers
# np.random.randn(2,3) #Returns a standard normal distribution with mean 0 and variance 1
# np.random.randint(0, 101, 10) #Returns 10 random Integers between 0 and 101
# np.random.randint(0, 101, (4,5)) #Returns 10 random Integers between 0 and 101 in a 4x5 array
# np.random.seed(42) #Returns a set of numbers in .rand
# np.random.rand(4) #Result of .seed
# arr = np.arange(0,25)
# arr.reshape(5,5) #Reshapes array into a 5x5 array
# randarr = np.random.randint(0, 101, 10)
# randarr.max() #Returns max Number in array
# randarr.min() #Returns min Numberin array
# randarr.argmax() #Returns index of max number
# randarr.argmin() #Returns index of min number
# randarr.dtype #Returns type of array
# arr.shape #Returns shape of the array

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
# myindex = ['Canada', 'Uganda', 'Australia']
# mydata = ['CA', 'UG', 'AU']
# myser = pd.Series(data=mydata, index=myindex)
# print(myser)
# print(myser[0])
# print(myser['Uganda'])
# print(myser.keys())

# ages = {'Jean-Paul':5,'Francois':10,'Genevieve':50}
# pd.Series(ages)

# mycompanies = ['Microsoft', 'Google', 'Nvidia']
# sales = [0.5, 2, 10]

# mycompanies1 = ['Microsoft', 'Google', 'Nvidia']
# sales1 = [2, 3, 20]

# sales_q1 = pd.Series(index=mycompanies,data=sales)
# sales_q2 = pd.Series(index=mycompanies1,data=sales1)
# total_sales = sales_q1.add(sales_q2,fill_value=0)
# print(total_sales)

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
# df = pd.read_csv('fake_reg.csv')
# print(df.head())
# sns.pairplot(df)
# X = df[['feature1','feature2']].values
# y = df['price'].values
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# model = Sequential([
#     Dense(unit=4,activation='relu'),
#     Dense(unit=2,activation='relu'),
#     Dense(unit=1)
#     ])

# model = Sequential()
# model.add(Dense(4,activation='relu')) #unit = neuron
# model.add(Dense(4,activation='relu'))
# model.add(Dense(4,activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='rmsprop',loss='mse')

# model.fit(x=X_train,y=y_train,epochs=250) #Train/Fit Model
#epochs=number of times it iterates

# loss_df = pd.DataFrame(model.history.history)
# loss_df.plot()
# plt.show()

# model.evaluate(X_train,y_train,verbose=0)
# test_predictions = model.predict(X_test)
# test_predictions = pd.Series(test_predictions.reshape(300,))
# pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
# pred_df = pd.concat([pred_df,test_predictions],axis=1)
# pred_df.columns = ['Test True Y', 'Model Predictions']
# sns.scatterplot(x='Test True Y',y='Model Predictions',data=pred_df)
# mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])
# mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])
# df.describe()
# print(pred_df)

#To save a model
#model.save('model_name')
#model_name = load_model('model_name')



# df = pd.read_csv('kc_house_data.csv')
# df.isnull() #returns back true if something is null
# df.isnull().sum() #returns sum of nulls
# df.head()
# df.describe().transpose()

# plt.figure(figsize=(10,6))
# sns.histplot(df['price'])
# plt.show()

# sns.countplot(df['bedrooms'])
# plt.show()

# df.corr()
# print(df.corr()['price'].sort_values())
# plt.figure(figsize=(10,5))
# sns.scatterplot(x='price',y='sqft_living',data=df)
# plt.show()

# plt.figure(figsize=(10,6))
# sns.boxplot(x='bedrooms',y='price',data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='price',y='long',data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='price',y='lat',data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long',y='lat',data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long',y='lat',data=df,hue='price')
# plt.show()

# print(df.sort_values('price', ascending=False).head(20))
# len(df) #num of houses
# len(df) * 0.01 #top 1%

# non_top_1_percent = df.sort_values('price', ascending=False).iloc[216:]
# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long',y='lat',data=non_top_1_percent,edgecolor=None,hue='price',alpha=0.2,palette='Accent')
# plt.show()

# sns.boxplot(x='waterfront',y='price',data=df)

# df = df.drop('id',axis=1)
# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].apply(lambda date: date.year)
# df['month'] = df['date'].apply(lambda date: date.month)
# df.drop('date',axis=1)
# print(df.head())

# plt.figure(figsize=(10,5))
# sns.boxplot(x='month',y='price',data=df)
# print(df.groupby('month').mean()['price'])
# print(df['zipcode'].value_counts())
# print(df['sqft_basement'].value_counts())
# df = df.drop('zipcode',axis=1)
# plt.show()

# X = df.drop('price',axis=1).values
# y = df['price'].values
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# model = Sequential()
# print(X_train.shape) #second param is number of nodes which is 19
# model.add(Dense(19,activation='relu'))
# model.add(Dense(19,activation='relu'))
# model.add(Dense(19,activation='relu'))
# model.add(Dense(19,activation='relu'))

# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
# model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)
# print(model.history.history)
# print(pd.DataFrame(model.history.history))

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

# df = pd.read_csv('cancer_classification.csv')
# print(df.describe().transpose())

# sns.countplot(x='benign_0__mal_1',data=df)
# plt.show()

# print(df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar'))
# plt.show()

# plt.figure(figsize=(12,12))
# sns.heatmap(df.corr())
# plt.show()

# X = df.drop('benign_0__mal_1',axis=1).values
# y = df['benign_0__mal_1'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = Sequential()
# model.add(Dense(30,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(15,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam')
# early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
# model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

# predictions = (model.predict(X_test) > 0.5).astype(int)
# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test,predictions))

################################################################################################

#Convulutional Neural Networks with TensorFlow


# (x_train,y_train), (x_test,y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train)
# single_image = x_train[0]
# print(single_image.shape)
# plt.imshow(single_image)

# y_example = to_categorical(y_train)
# print(y_example.shape)
# print(y_example[0])

# y_cat_test = to_categorical(y_test,num_classes=10)
# y_cat_train = to_categorical(y_train,10)
# x_train = x_train/255 #255 is single_image.max() value
# x_test = x_test/255
# scaled_image = x_train[0]
# print(scaled_image.max()) #scaled_image will be equaled to 1

# print(x_train.shape)
# x_train = x_train.reshape(60000,28,28,1) #batch_size,width,height,color_channels
# x_test = x_test.reshape(10000,28,28,1)

# model = Sequential()
# model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),input_shape=(28,28,1),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))

# #Softmax because multiclass problem
# model.add(Dense(10,activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_loss',patience=1)
# model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

# metrics = pd.DataFrame(model.history.history)
# metrics[['loss','val_loss']].plot()
# plt.show()

# metrics[['accuracy','val_accuracy']].plot()
# plt.show()

# model.evaluate(x_test,y_cat_test,verbose=0)

# predictions = model.predict_classes(x_test)
# print(classification_report(y_test,predictions))
# print(confusion_matrix(y_test,predictions))
# plt.figure(figsize=(10,6))
# sns.heatmap(confusion_matrix(y_test,predictions))
# plt.show()
# my_num = x_test[0]
# plt.imshow(my_num.reshape(28,28))
# plt.show()
# model.predict_classes(my_num.reshape(1,28,28,1))




# (x_train,y_train), (x_test,y_test) = cifar10.load_data()

# print(x_train.shape)
# print(x_train[0].shape)

# plt.imshow(x_train[6])
# plt.show()

# print(x_train[6].max())

# x_train = x_train/255
# x_test = x_test/255

# y_cat_train = to_categorical(y_train,10)
# y_cat_test = to_categorical(y_test,10)

# model = Sequential()
# model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),input_shape=(32,32,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(filters=16,kernel_size=(4,4),strides=(1,1),input_shape=(32,32,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(256,activation='relu'))

# model.add(Dense(10,activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model.summary())

# early_stop = EarlyStopping(monitor='val_loss',patience=2)
# model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

# metrics = pd.DataFrame(model.history.history)
# metrics[['loss','val_accuracy']].plot()
# plt.show()

# metrics[['loss','val_loss']].plot()
# plt.show()

# print(model.evaluate(x_test,y_cat_test,verbose=0))

# predictions = model.predict_classes(x_test)
# print(classification_report(y_test,predictions))

# print(confusion_matrix(y_test,predictions))

# plt.figure(figsize=(10,16))
# sns.heatmap(confusion_matrix(y_test,predictions))
# plt.show()

# my_image = x_test[0]
# plt.imshow(my_image)
# plt.show()
# model.predict_classes(my_image.reshape(1,32,32,3))




# data_dir = 'C:\\Users\\prodo\\Downloads\\cell_images\\cell_images'
# print(os.listdir(data_dir))

# test_path = data_dir + '\\test\\'
# train_path = data_dir + '\\train\\'
# print(os.listdir(test_path))
# print(os.listdir(train_path))

# print(os.listdir(train_path+'parasitized')[0])

# dim1 = []
# dim2 = []
# for image_filename in os.listdir(test_path+'uninfected'):
#     img = imread(test_path+'uninfected\\'+image_filename)
#     d1,d2,colors = img.shape
#     dim1.append(d1)
#     dim2.append(d2)

# sns.joinplot(dim1,dim2)

# print(np.mean(dim1))
# print(np.mean(dim2))
# image_shape = (130,130,2)

# image_gen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')

# model = Sequential()
# model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam')
# print(model.summary())

# early_stop = EarlyStopping(monitor='val_loss',patience=2)
# train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],
#                                                 color_mode='rgb',batch_size=16,class_mode='binary')
# test_image_gen = image_gen.from_flow_directory(test_path,target_size=image_shape[:2],
#                                                color_mode='rgb',batch_size=16,class_mode='binary',
#                                                shuffle=False)

# results = model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,callbacks=[early_stop])
# model.evaluate_generator(test_image_gen)
# prediction = model.predict_generator(test_image_gen)

################################################################################################

