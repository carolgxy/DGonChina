# DG on China

## Abstract
The demands of socioeconomic development frequently lead to large-scale population migration among cities. While complex network and population migration algorithms have been employed to evaluate this phenomenon, predicting the future shift of urban networks has remained challenging. In this study, we expend the conventional two-dimensional perception of urban structure, projecting geographic information of cities into a high-dimensional future dimension to forecast changes in the network structure with deep learning algorithms. Using the population migration data from 362 Chinese cities, we employed multivariate and non-linear layers to construct a deep learning model that exhibits good geographic and temporal generalization across major metropolitan regions in China, enabling us to forecast the urban network for the year 2025. The result shows that the urban network becomes more equitable and less concentrated in a few dominant cities. This shift suggests a more balanced distribution of resources, opportunities, and development across the urban agglomerations. Understanding the urban structure from the lens of future mobile networks offers deeper insight and perception of its future dimensional nature. By embracing this paradigm shift, we can retain knowledge about urban dynamics and pave the way for more effective urban management.

Keywords: Urban network; Urban population migration; Mobility prediction; Deep learning model; Urban agglomerations

## Setup
`DG_on_China.yaml` is recommend to setup the python 3.7 environment.

## Running
The code is organized into 3 section in JupyterNotebook. 
- **Section 1 "Initialization"** ：Initializes some variables that must be run.
- **Section 2 "Training"** ：It will complete the training of the DG model. the relevant data has be cleaned and stored in the `data` folder, and the training results will be saved in the `result` folder.
- **Section 3 "Prediction"** ：It is to make predictions using the trained model. Note that the author has uploaded a trained model, so you can directly run the code in the section 3 to see the prediction results.

## Citation
Xinyue Gu, Xingyu Tang, Tong Chen, Xintao Liu, Predicting the network shift of large urban agglomerations in China using the deep-learning gravity model: A perspective of population migration, Cities, Volume 145, 2024, 104680, ISSN 0264-2751, https://doi.org/10.1016/j.cities.2023.104680.
