import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import CSVLogger

# training_file = 'D:\IoTVaUngDung\TaiLieu\KDDTrain+.arff'
# testing_file = 'D:\IoTVaUngDung\TaiLieu\KDDTest+.arff'
# def getDataAfterPreprocessing(file_dir, set_file_name):
#     csv_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in'
#                  , 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
#                  'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
#                  'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
#                  'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class']
#     dict_data = []
#     f = open(file_dir, mode = 'r')
#     lines = []
#
#     for line in f:
#         lines.append(line.replace('\n', ''))
#     for i in range(44):
#         lines.remove(lines[0])
#
#     for line in lines:
#         words = line.split(',')
#         dict = {}
#         for i in range(len(words)):
#             dict[csv_columns[i]] = words[i]
#         dict_data.append(dict)
#
#     dict_data = pd.DataFrame.from_dict(dict_data)
#     encoder = LabelEncoder()
#     for name in dict_data.columns:
#         if name in ['protocol_type','service','flag', 'class']:
#             dict_data[name] = encoder.fit_transform(dict_data[name])
#
#     dict_data.to_csv (set_file_name, index = False, header=True)
#
# getDataAfterPreprocessing(testing_file, 'Test.csv')
#Chay chuong trinh
traindata = pd.read_csv('Train.csv')
testdata = pd.read_csv('Test.csv')
X = traindata.iloc[:,0:41]
Y = traindata.iloc[:,41]
C = testdata.iloc[:,41]
T = testdata.iloc[:,0:41]
# print(C)

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
# print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
# print(testT[0:5,:])

y_train = np.array(Y)
y_test = np.array(C)
#
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))
# print(X_train.shape)
batch_size = 32
# 1. define the network
model = Sequential()
model.add(LSTM(4, input_dim=41))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# print(model.get_config())

# try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# checkpointer = callbacks.ModelCheckpoint(filepath="results/checkpoint-{epoch:02d}.h5", verbose=1, save_best_only=True, monitor='val_accuracy',mode='max')
# csv_logger = CSVLogger('results/training_set_iranalysis.csv',separator=',', append=False)
# model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
# model.save("results/lstm1layer_model.h5")

# loss, accuracy = model.evaluate(X_test, y_test)
# print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
model = keras.models.load_model('results/lstm1layer_model.h5')
y_pred = model.predict(X_test)
for i, x in enumerate(y_pred):
    if x < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
print(y_pred)
np.savetxt('lstm1predicted.txt', y_pred, fmt='%01d')