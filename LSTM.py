import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import CSVLogger
from keras import models
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
#     # dict_data.to_csv (set_file_name, index = False, header=True)
#
# getDataAfterPreprocessing(testing_file, 'Test.csv')
#Chay chuong trinh
traindata = pd.read_csv('Train3.csv')
testdata = pd.read_csv('Test.csv')
X = traindata.iloc[:,0:41]
Y = traindata.iloc[:,41]
C = testdata.iloc[:,41]
T = testdata.iloc[:,0:41]
# print(Y)

#SelectKBest
# bestfeatures = SelectKBest(score_func=chi2, k=41)
# fit = bestfeatures.fit(X, Y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Specs', 'Score']
# print(featureScores.nlargest(41, 'Score'))
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
# print(trainX[0:5,:])
#
scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
# print(testT[0:5,:])

y_train1 = np.array(Y)
y_test1 = np.array(C)
print(y_train1)
#
y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)
# print(y_train)
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))
# print(X_train.shape)
batch_size = 32
# 1. define the network
model = Sequential()
model.add(LSTM(4,input_dim=41))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))
# print(model.get_config())

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# checkpointer = callbacks.ModelCheckpoint(filepath="results/checkpoint-{epoch:02d}.h5", verbose=1, save_best_only=True, monitor='val_accuracy',mode='max')
# csv_logger = CSVLogger('results/training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, epochs=5)
# , validation_data=(X_test, y_test)
# ,callbacks=[checkpointer,csv_logger]
# model.save("results/lstm1layer_model.h5")

# loss, accuracy = model.evaluate(X_test, y_test)
# print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# model = models.load_model('results/lstm1layer_model.h5')
# y_pred = model.predict(X_test)
#
# TP = 0
# TN = 0
# FP = 0
# FN = 0
#
# for i, x in enumerate(y_pred):
#     if x < 0.5:
#         y_pred[i] = 0
#         if C[i] == 0:
#             TN = TN + 1
#         else:
#             FN = FN + 1
#     else:
#         y_pred[i] = 1
#         if C[i] == 0:
#             FP = FP + 1
#         else:
#             TP = TP + 1
# # print(y_pred)
# # np.savetxt('lstm1predicted.txt', y_pred, fmt='%01d')
# Accuracy = (TP + TN)/(TP + TN + FP + FN)
# TPR = TP / (TP + FN)
# PPV = TP / (TP + FP)
# NPV = TN / (TN + FN)
# FNR = FN / (FN + TP)
# FPR = FP / (FP + TN)
# FDR = FP / (FP + TP)
# FOR = FN / (FN + TN)
# F1_score = 2*TP/(2*TP + FP + FN)
# print('Accuracy: ' + str(round(Accuracy*100,2)) + '\n' + 'TPR: ' + str(round(TPR*100,2)) + '\nPPV: ' + str(round(PPV*100,2)) + '\nNPV: ' + str(round(NPV*100,2))
#       + '\nFNR: ' + str(round(FNR*100,2)) + '\nFPR: ' + str(round(FPR*100,2)) + '\nFDR: ' + str(round(FDR*100,2)) + '\nFOR: '
#       + str(round(FOR*100,2)) + '\nF1 score: ' + str(round(F1_score*100,2)))