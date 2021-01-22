Source code for paper "On Filter Generalization for Music Bandwidth Extension Using Deep Neural Networks", 
Serkan Sulun, Matthew E. P. Davies, 2020. 
https://arxiv.org/abs/2011.07274v2


To cite:

```S. Sulun and M. E. P. Davies, "On Filter Generalization for Music Bandwidth Extension Using Deep Neural Networks," in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2020.3037485.```

Required Python libraries: Numpy, Scipy, Pytorch, Requests, tqdm

For CUDA 11.0 you can also run `pip install -r requirements.txt`

To download datasets:
Run "get_datasets.py" within folder "datasets".
Only downloads DSD100, since MedleyDB requires special permission.

To download pre-trained models:
Run "get_models.py --model [modelname]" within the folder "output/models".

config.py specifies hyperparameters, command-line arguments and general configuration.

Main script: "run.py" within folder "src"

Tested with Python 3.7.9, Numpy 1.19.4, Scipy 1.6.0, Pytorch 1.7.1


Acknowledgements

Serkan Sulun receives the support of a fellowship from ”la Caixa” Foundation (ID 100010434), with the fellowship code LCF/BQ/DI19/11730032.
This work is also funded by national funds through the FCT - Foundation for Science and Technology, I.P., within the scope of the project CISUC - UID/CEC/00326/2020 and by European Social Fund, through the Regional Operational Program Centro 2020 as well as by Portuguese National Funds through the FCT - Foundation for Science and Technology, I.P., under the project IF/01566/2015.
