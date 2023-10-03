# DG on China

## Abstract
Due to the demands of socioeconomic development, large-scale population migration
between cities frequently occurs. While complex network and population migration
algorithms have been employed to evaluate this phenomenon, predicting the future
shift of urban networks has remained challenging. In this study, we propose an
extension of the conventional two-dimensional perception of urban structure, projecting
geographic information of cities into a high-dimensional future dimension to forecast
changes in the network structure with deep learning algorithms. Based on the
population migration data from 362 Chinese cities, we employed multivariate and nonlinear
layers to construct a training model that exhibits good geographic and temporal
generalization across major metropolitan regions in China, enabling us to forecast the
urban network for the year 2025. The result shows that the urban network becomes
more equitable and less concentrated in a few dominant cities. This shift suggests a
more balanced distribution of resources, opportunities, and development across the
urban agglomerations. Understanding the urban structure from the lens of future
mobile networks offers deeper insight and perceive its future dimensional nature. By
embracing this paradigm shift, we can retain a wealth of knowledge about urban
dynamics and pave the way for more effective urban management.

## Setup
`DG_on_China.yaml` is recommend to setup the python 3.7 environment.

## Running
The code is organized into 3 section in JupyterNotebook. 
- **Section 1 "Initialization"** ：Initializes some variables that must be run.
- **Section 2 "Training"** ：It will complete the training of the DG model. the relevant data has be cleaned and stored in the `data` folder, and the training results will be saved in the `result` folder.
- **Section 3 "Prediction"** ：It is to make predictions using the trained model. Note that the author has uploaded a trained model, so you can directly run the code in the section 3 to see the prediction results.
