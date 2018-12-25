# pyriodogram


Python re-implementation of Fast calculation of the Lomb-Scargle periodogram using nonequispaced fast Fourier transforms [1]. Original code can be found at [2].

[1] https://www.aanda.org/articles/aa/pdf/2012/09/aa19076-12.pdf

[2] http://corot-be.obspm.fr/code.php

## Requisites

This package uses Lightweight non-uniform Fast Fourier Transform in Python [3] for the computation of the ndft. 
The nfft package can be installed directly from the Python Package Index:

pip install nfft

[3] https://github.com/jakevdp/nfft

## Example

Extracted ndft features for an astronomical source in the plasticc dataset [4] with object_id 612.

![Alt text](example.png?raw=true "NDFT features for plasticc dataset with object_id 612")

[4] https://www.kaggle.com/c/PLAsTiCC-2018
