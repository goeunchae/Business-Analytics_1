# Business-Analytics_1, Dimensionality Reduction 

## Dataset 
### Regression Dataset

California House Price

Data Link: https://www.kaggle.com/datasets/shibumohapatra/house-price
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/EDA_housing_raw.PNG)

[->Data EDA](https://github.com/goeunchae/Business-Analytics_1/tree/main/EDA_housing.ipynb)


### Classification Dataset

Wine Quality Dataset

Data Link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/EDA_wine_raw.PNG)

[->Data EDA](https://github.com/goeunchae/Business-Analytics_1/tree/main/EDA_wine.ipynb)

## 1-2 Supervised Variable Selection 

Conduct supervised variable selection in 3-way with regression dataset (California house price). 
Overall R-squared is not that great but we still see the difference between among seleciton methods.


### Forward Selection 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_forward_selection.png)


We got 'ocean_<1H OCEAN', 'population', 'total_rooms', 'total_bedrooms', 'housing_median_age', 'households', 'ocean_INLAND', 'longitude', 'latitude' as our final variables. It seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 

### Backward Elimination
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_backward_elimination.png)


We got 'ocean_<1H OCEAN', 'ocean_INLAND', 'ocean_ISLAND', 'ocean_NEAR BAY', 'ocean_NEAR OCEAN', 'longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population' as our final variables. Variables releated to ocean vicinity are all selected, and population, number of rooms are also importance as forward selection. It also seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 


### Stepwise Selection
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_stepwise_selection.png)

We got 'ocean_<1H OCEAN', 'population', 'total_rooms', 'total_bedrooms', 'housing_median_age', 'households', 'ocean_INLAND', 'longitude', 'latitude' as our final variables. Selected variables are similar to those from forward selection and backward elimination. It also seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 

Generally, about 9-10 variables are selected for regression and the most importance variables is 'ocean_<1H OCEAN'. It means closer to the ocean, higher the housing price in California which is reasonable. 


## 1-2 Genetic Algorithm

Conduct genetic algorithm with classification dataset (wine quality dataset). Compare the result with feature importance based on random forest. 


### GA Raw Result 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_GA_results.PNG)

### GA Result 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_GA_wine.PNG)

With genetic algorithm, we found the best variable set [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]. According to the result, the best varialbes for wine quality classification are volatile acidity, residual sugar, chlorides, total sulfur dioxide, pH, alcohol. [1, 3, 4, 6, 8, 10]

### Random Forest Importance Score 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_feature_importance.png)

According to above figure, feature 1, feature 6, feature 10 are remarkably important -> volatile acidity, total sulfur dioxide, alcohol. 
Selected variables with GA also have high feature importances in random forest classificaiton. With this result, we can say that GA worked well for selecting variables. 


## 1-3 Principal Component Analysis (PCA) 
### Explained Variance 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_percentage_of_explained_variance.png)



### PCA 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_pca_results.png)

### PCA with sklearn  
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_pca_sklearn.png)


## 1-3 Multi-Dimensional Scaling (MDS) 
### MDS
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_MDS_sklearn.png)

## 1-4 Isometric Feature Mapping (ISOMAP) 
### ISOMAP
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_isomap_results.png)

### ISOMAP with sklearn
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_isomap_results_with_sklearn.png)

## 1-4 Locally Linear Embedding (LLE) 
### LLE
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results.png)

## 1-4 t-distributed Stochastic Neighbor Embedding (t-SNE)  
### t-SNE
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE_results.png)

