# iasi-atmosphere
Infrared Atmospheric Sounding Inteferometer (IASI) data set repository. For loading data and training CNNs for prediction of atmospheric profiles, like temperatures and water pressure.Repository for the publications "Statistical retrieval of atmospheric profiles with deep convolutional neural networks"


### NB! As I ended my academic position before finishing this code, it is not fully functional. The code was meant to run Conv. Neural Networks on IASI atmospheric profiles, and is provided as is for inspirational use.

In utils are to decomposition functions used to create a decomposition matrix. This matrix dot product with a data matrix of IASI profiles can reduce the dimensions of IASI data from the 4699 spectral channels it has in the dataset to some smaller dimension that a CNN can handle.