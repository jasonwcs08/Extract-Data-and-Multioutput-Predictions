import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from numpy import argmax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

#read csv file
df = pd.read_csv("LMNO_processed.csv")


#Create Correlation Heatmap
df.corr()
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

# save heatmap as .png file
# dpi - sets the resolution of the saved image in dots/inches
# bbox_inches - when set to 'tight' - does not allow the labels to be cropped
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


#Select X and y values
X = df[['Li', 'Mn', "Ni", 'O']]
y_reg = df[["formation_energy_per_atom"]]

print (y_reg[0:5])

#Transform string to encoded label
lb = LabelEncoder()
y_class = lb.fit_transform(df[["crystal_system"]]) #0=cubic, 1=hexagonal, 2= monoclinic, 3=orthorhombic, 4= Tetragonal,5=triclinic, 6=trigonal

print (df[["crystal_system"]].head())
print (y_class[0:5])

#Number of Features
n_features = X.shape[1] #4 types of input features (Li, Mn, Ni and O elements)
print ("Number of Features:", n_features)

#Number of Class
n_class = len(lb.classes_) #7 types of crystal systems
print ("Number of Class in Crystal System:", n_class)


#Split Train and test data (X, y value for regression and y value for classification)
X_train, X_test, y_train, y_test, y_train_class, y_test_class = train_test_split(X, y_reg, y_class, test_size=0.3, random_state=36)

#Build model
visible = Input(shape=(n_features,))
hidden1 = Dense(100, activation='relu', kernel_initializer='he_normal')(visible)
hidden2 = Dense(50, activation='relu', kernel_initializer='he_normal')(hidden1)
hidden3 = Dense(30, activation='relu', kernel_initializer='he_normal')(hidden2)
hidden4 = Dense(10, activation='relu', kernel_initializer='he_normal')(hidden3)

#Regression output layer
out_reg = Dense(1, activation='linear')(hidden4)
#Classification output layer
out_clas = Dense(n_class, activation='softmax')(hidden4)
#Define model
model = Model(inputs=visible, outputs=[out_reg, out_clas])
#Compile the keras model
model.compile(loss=['mse','sparse_categorical_crossentropy'], optimizer='adam')

#Save model weights with minimum validation loss
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='min', save_freq='epoch')

#Stop training when a monitored metric has stopped improving.
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

#Fit the keras model on the dataset
model.fit(X_train, [y_train,y_train_class], validation_data=(X_test, [y_test,y_test_class]),epochs=200, batch_size=5, callbacks=[checkpoint, early])


#Make predictions on test set
yhat1, yhat2 = model.predict(X_test)

#Calculate error for regression model
error = mean_absolute_error(y_test, yhat1)
print('Mean Absolute Error for Regression: %.3f' % error)

#Evaluate accuracy for classification model
yhat2 = argmax(yhat2, axis=-1).astype('int')
acc = accuracy_score(y_test_class, yhat2)
print('Classification Accuracy: %.3f' % acc)


#Generate Confusion Matrix Map
cf_matrix = confusion_matrix(yhat2, y_test_class)
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted Output')
ax.set_ylabel('Actual Output')
ax.set_title('Confusion Matrix')
plt.show()


#Do prediction with trained model
pred1, pred2 = model.predict([[0.4, 0.3, 0.2, 0.1]])

print ("Predicted Formation Energy:", pred1)
crystal = lb.inverse_transform(argmax(pred2, axis=-1))
print ("Predicted Crystal:", crystal)