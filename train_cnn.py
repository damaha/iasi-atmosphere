import json, os
import numpy as np
import pandas as pd

from data_load import get_datafiles
from generators import iasi_generator
from models import *

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

### Parameters ###
params = {"ModelName" : "models/model_firsttry", 
          "ModelType" : simple_model_1,
          "n_comp" : 128,
          "f_spec" : 5,
          "batchsize" : 16,
          "epochs" : 52,
          "period_weights" : 10,
          "seed" : 18374,
          "outputs" : np.arange(47, 137).astype(int),
          "path" : ".../IASI/data_v3/",
          "train_code" : open("train_cnn.py", "r").read(),
          "model_code" : open("models.py", "r").read(),
          "generator_code" : open("generators.py", "r").read()}
np.random.seed(params['seed'])


### Split data and make generators ###
train_files, valid_files = get_datafiles(params["path"])

X_shape = np.load(valid_files[0][0]).shape
Y_shape = np.load(valid_files[0][1]).shape

indexs = [None, params["outputs"]]

dc = json.load(open("scaling_coeffs.json"))

train_generator = iasi_generator(train_files, batch_size=params['batchsize'], selected_channels=indexs, norm_coeffs=[dc['mean'], dc['variance']])
valid_generator = iasi_generator(valid_files, batch_size=params['batchsize'], selected_channels=indexs, norm_coeffs=[dc['mean'], dc['variance']])

model = params["ModelType"](X_shape, params["outputs"].size, n_comp=params["n_comp"], f_spec=params["n_comp"])
if os.path.isfile(params["ModelName"]+"_config.json") and os.path.isfile(params["ModelName"]+".h5"):
    if os.path.isfile(params["ModelName"]+"_history.json"):
        json_hist = json.load(open(params["ModelName"]+"_history.json","r"))
    else:
        json_hist = {'loss':[],'val_loss':[]}

    json_str = json.load(open(params["ModelName"]+"_config.json","r"))
    model = model_from_json(json_str)
    model.load_weights(params["ModelName"]+".h5")
    log = np.genfromtxt(params["ModelName"]+".log", delimiter=",",skip_header=1)
    e_init = int(log[-1,0] // params["period_weights"] * params["period_weights"])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    print("Continuing training process from epoch %d" % e_init)
else:
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    json_string = model.to_json()
    json.dump(json_string, open(params["ModelName"]+"_config.json", "w"))
    model_object = model
    e_init = 0
    json_hist = {'loss':[],'val_loss':[]}

nb_gpus = len(get_available_gpus())
if nb_gpus > 1:
    m_model = multi_gpu_model(model, gpus=nb_gpus)
    m_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model_object = m_model
else:
    model_object = model

import time
start = time.time()
history = model_object.fit_generator(train_generator,
                                     callbacks=[ModelCheckpoint(params["ModelName"].split('/')[-1]+".{epoch:02d}.h5",
                                                                monitor='loss',
                                                                verbose=1,
                                                                period=params["period_weights"],
                                                                save_weights_only=True),
                                                # TensorBoard(log_dir='tmp/logs/'),
                                                CSVLogger(params["ModelName"]+'.log'),
                                                ReduceLROnPlateau(monitor='loss', 
                                                                  factor=0.2,
                                                                  patience=5, 
                                                                  min_lr=0.0001,
                                                                  min_delta=0.001)],
                                     validation_data=valid_generator, 
                                     epochs=params["epochs"], 
                                     max_queue_size=5, 
                                     verbose=2, 
                                     initial_epoch=e_init)
seconds = time.time()-start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print('It took %d:%d:%02d:%02d to train.' % (d, h, m, s))

model.save_weights(params["ModelName"]+".h5")


import socket
dct = history.history

dct['loss'] = json_hist['loss']+dct['loss']
dct['val_loss'] = json_hist['val_loss']+dct['val_loss']

dct["number_of_gpus"] = nb_gpus
dct["hostname"] = socket.gethostname()
dct["training_files"] = str(train_files)
dct["test_files"] = str(valid_files)
dct["training_time"] = '%d:%d:%02d:%02d' % (d, h, m, s)
dct.update(params)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

json.dump(dct, open(params["ModelName"]+"_history.json", 'w'), cls=NpEncoder)