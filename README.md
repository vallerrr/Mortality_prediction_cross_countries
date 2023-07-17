# Death Prediction: Holistically and Precisely with Explainable AI 
In this chapter, we propose a new predictive research paradigm with Machine Learning (ML) for learning health topics from a social viewpoint, which is comprehensive and bridges the gap between prediction and explanation with Explainable Artificial Intelligence algorithms. Specifically, we assess the contributions of various social risk factors at both individual levels using explainable AI and on a more holistic scale through the application of the 'leave one out' concept. We additionally devote considerable effort to addressing methodological concerns in predictive precision while executing ML algorithms, including quantifying the effects of seed selection and variable quantity impact, issues that have been a recurring focus in quantitative sociological research.

# Research Route Map 
![Research Route Map](https://github.com/vallerrr/OX_thesis/blob/main/graphs/intro/FlowChart.png)

- Predictive:
  Apply Super Learners to all four datasets and present their predictive performances. Results of LightGBM and Logistic Regression are also provided for comparisons.

- Holistic:  
  1. Employ SHAP to deconstruct LightGBM and assess risk importance at the individual risk factor level with HRS, SHARE and Combined dataset .
  
  2. Develop a ‘leave-one-out’ algorithm to explore predictive contributions using LightGBM at the domain level for all datasets.


- Precise: 

  Splitting the entire dataset into training and testing sets.
  
     1. Iteratively subset the training set with increasing proportions(30% - 100%).
  
     2. Iteratively and increasingly subset the risk factors (one to 26). 
  Then train the model on these subsets to predict the same test set.
  
  Training the same model on 10000 instances of train-test splitting with different seed values and compare the predictive performances.

![image](https://github.com/vallerrr/OX_thesis/assets/36286934/6c61ed62-77b4-448d-acb3-e9347d44613e)

# Project Folder Map 
![Project Folder Map](https://github.com/vallerrr/OX_thesis/blob/main/graphs/intro/ProjectMap.png)

Tips:

- Datasets are unavailable in this repo. Access to HRS, ELSA and SHARE can be received upon separate application to the data providers, information can be found through the [Gateway to Aging Data](https://g2aging.org/downloads)
- All the Data analysing codes are organised in the **main.ipynb** file, which invokes functions wrapped in the src folder 

