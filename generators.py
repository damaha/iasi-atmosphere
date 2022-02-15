import numpy as np
import keras
from osgeo import gdal

    
class iasi_generator(keras.utils.Sequence):
    """Class for keras data generation on IASI dataset."""

    def __init__(self, files, batch_size=32, selected_channels=None, shuffle=True, dim_red=None, meta=False, norm_coeffs=None):
        """'Initialization
        batch_size          : Size of batch to be returned
        files               : Should contain list of lists e.g [ [input_file1, target_file1], [input_file2, target_file2], ...]
                              or like [ [input_file1, target_file1, meta_file1], ...]
        shuffle             : True - Shuffels rows in files,
        dim_red = dim_red   : None or decomposition matrix of spectral dimension
        on_epoch_end()      : 
        selected_channels    : list of None and arrays. if element is an array, it contains indicies along the last dimension.
                              this is to perform exclusion of some bands or areas in images.

        """
        self.batch_size = batch_size
        self.files = files         
        self.shuffle = shuffle
        self.dim_red = dim_red      
        self.on_epoch_end()
        self.norm_coeffs = norm_coeffs
        if selected_channels:
            self.selected_channels = selected_channels
        else:
            self.selected_channels = [None]*len(files[0])
        self.shapes = []

        if self.selected_channels:
            for i, (el,ind) in enumerate(zip(files[0],self.selected_channels)):
                if isinstance(ind, np.ndarray):
                    self.shapes.append(np.load(el)[..., ind].shape)
                else:
                    self.shapes.append(np.load(el).shape)
        else:
            for el in files[0]:
                    self.shapes.append(np.load(el).shape)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return(int(np.floor(len(self.files) / self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [self.files[k] for k in indexes]

        # Generate data
        data = self.__data_generation(batch_files)

        return( data)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):
        """Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)"""

        # Create Empty arrays
        data_list = []
        for shp in self.shapes:
            data_list.append(np.empty((self.batch_size, *shp)))
        
        # Load data
        for i in range(self.batch_size):            
            for j, (el, ind) in enumerate(zip(batch_files[i], self.selected_channels)):
                if isinstance(ind, np.ndarray):
                    data_list[j][i] = np.load(el)[..., ind].astype('float32')
                else:
                    data_list[j][i] = np.load(el).astype('float32')
    
            if self.dim_red:
                data_list[0][i] = np.dot(data_list[0].reshape((self.shapes[0]*self.shapes[1],self.shapes[2])), self.dim_red)
            elif self.norm_coeffs:
                data_list[0][i] = (data_list[0][i] - np.array(self.norm_coeffs[0]).reshape((1,1,1,len(self.norm_coeffs[0]))) ) / np.array(self.norm_coeffs[1]).reshape((1,1,1,len(self.norm_coeffs[1])))

            # if data_aug:
            #     # add code for data augmentation here...

        return(data_list)