import os
import numpy as np

from generators import iasi_generator

def get_datafiles(path,
                  split_on = "dates",
                  split_frac = 0.2,
                  train_dates = ['20130801', '20130817', '20130717', '20130731', '20130816', 
                                 '20130831', '20130901', '20130916', '20130917', '20130930', 
                                 '20131001', '20131016', '20131017', '20131031', '20131101', 
                                 '20131116', '20131117', '20131130', '20131201', '20131216', 
                                 '20131217', '20131231'],
                  valid_dates = ['20140101', '20140116', '20140117']):

    if split_on == "dates":
        train_files = []
        for dat in train_dates:
            train_files.append([el for el in os.listdir(path+dat+'/')])
        train_files = np.unique([path+fil[:8]+'/'+'_'.join(fil.split('_')[:2]) for fil in np.hstack(train_files)])
        valid_files = []
        for dat in valid_dates:
            valid_files.append([el for el in os.listdir(path+dat+'/')])
        valid_files = np.unique([path+fil[:8]+'/'+'_'.join(fil.split('_')[:2]) for fil in np.hstack(valid_files)])

    train_files = [[fil+'_data.npy',fil+'_TP.npy'] for fil in train_files]
    valid_files = [[fil+'_data.npy',fil+'_TP.npy'] for fil in valid_files]

    return(train_files,valid_files)