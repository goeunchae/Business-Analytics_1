# Business-Analytics_1, Dimensionality Reduction 
## Tutorial Purposes

**1.** Apply various algorithms of dimensionality reduction to real-world dataset. 

**2.** Understand how the algorithm works while implementing it.

**3.** Explain about the result from algorithms with the knowledge of data 

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

Selecting variables with forward selection with r-squared of linear regression. 


```
# Forward Selection 
def FC(df):
    variables = df.columns[:-2].tolist() 
    
    y = df['median_house_value'] ## response variable
    selected_variables = [] 
    sl_enter = 0.05
    
    sv_per_step = [] ## selected variables per step
    adjusted_r_squared = [] 
    steps = [] ## 스텝
    step = 0
    while len(variables) > 0:
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder) 
        
        for col in remainder: 
            X = df[selected_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y,X).fit()
            pval[col] = model.pvalues[col]
    
        min_pval = pval.min()
        if min_pval < sl_enter: 
            selected_variables.append(pval.idxmin())
            
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y,sm.add_constant(df[selected_variables])).fit().rsquared_adj
            adjusted_r_squared.append(adj_r_squared)
            sv_per_step.append(selected_variables.copy())
        else:
            break
    return selected_variables, steps, adjusted_r_squared, sv_per_step 
```


plotting selected variables at every steps. 

```
def result_pic(selected_variables, steps, adjusted_r_squared, sv_per_step):
    fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('white')
    
    font_size = 15
    plt.xticks(steps,[f'step {s}\n'+'\n'.join(sv_per_step[i]) for i,s in enumerate(steps)], fontsize=12)
    plt.plot(steps,adjusted_r_squared, marker='o')
        
    plt.ylabel('Adjusted R Squared',fontsize=font_size)
    plt.grid(True)
    plt.show()
    print(selected_variables)
```
    


![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_forward_selection.png)


We got 'ocean_<1H OCEAN', 'population', 'total_rooms', 'total_bedrooms', 'housing_median_age', 'households', 'ocean_INLAND', 'longitude', 'latitude' as our final variables. It seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 

### Backward Elimination

Selecting variables with backward elimination with r-squared of linear regression. We defined sl_remove very small to prevent a step from not working.  


```
def BE(df):
    variables = df.columns[:-2].tolist() 
    
    y = df['median_house_value'] ## response variable
    selected_variables = variables ## every variable is chosen at the beginning
    sl_remove = 5e-50
    
    sv_per_step = [] ## selected variables per step
    adjusted_r_squared = [] 
    steps = []
    step = 0
    while len(selected_variables) > 0:
        X = sm.add_constant(df[selected_variables])
        p_vals = sm.OLS(y,X).fit().pvalues[1:] 
        max_pval = p_vals.max() 
        if max_pval >= sl_remove:
            remove_variable = p_vals.idxmax()
            selected_variables.remove(remove_variable)
    
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y,sm.add_constant(df[selected_variables])).fit().rsquared_adj
            adjusted_r_squared.append(adj_r_squared)
            sv_per_step.append(selected_variables.copy())
        else:
            break
    return selected_variables, steps, adjusted_r_squared, sv_per_step 
   ```

![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_backward_elimination.png)


We got 'ocean_<1H OCEAN', 'ocean_INLAND', 'ocean_ISLAND', 'ocean_NEAR BAY', 'ocean_NEAR OCEAN', 'longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population' as our final variables. Variables releated to ocean vicinity are all selected, and population, number of rooms are also importance as forward selection. It also seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 


### Stepwise Selection


Selecting variables with stepwise selection with r-squared of linear regression. sl_enter and sl_remove are 0.05. We have no selected variables at first like forward selection, and we also take out variables which affects negatively to r-squared like backward elimination.   


```
def SS(df):
    variables = df.columns[:-2].tolist() 
    y = df['median_house_value'] ## response variable
    selected_variables = [] 
    sl_enter = 0.05
    sl_remove = 0.05
    
    sv_per_step = [] ## selected variables per step
    adjusted_r_squared = [] 
    steps = [] 
    step = 0
    while len(variables) > 0:
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder) 
        for col in remainder: 
            X = df[selected_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y,X).fit()
            pval[col] = model.pvalues[col]
    
        min_pval = pval.min()
        if min_pval < sl_enter: 
            selected_variables.append(pval.idxmin())

            while len(selected_variables) > 0:
                selected_X = df[selected_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y,selected_X).fit().pvalues[1:]
                max_pval = selected_pval.max()
                if max_pval >= sl_remove:
                    remove_variable = selected_pval.idxmax()
                    selected_variables.remove(remove_variable)
                else:
                    break
            
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y,sm.add_constant(df[selected_variables])).fit().rsquared_adj
            adjusted_r_squared.append(adj_r_squared)
            sv_per_step.append(selected_variables.copy())
        else:
            break
    return selected_variables, steps, adjusted_r_squared, sv_per_step 
  ```
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_stepwise_selection.png)

We got 'ocean_<1H OCEAN', 'population', 'total_rooms', 'total_bedrooms', 'housing_median_age', 'households', 'ocean_INLAND', 'longitude', 'latitude' as our final variables. Selected variables are similar to those from forward selection and backward elimination. It also seems ocean vicinity, population, number of rooms are important variables to measure housing median price. 

Generally, about 9-10 variables are selected for regression and the most importance variables is 'ocean_<1H OCEAN'. It means closer to the ocean, higher the housing price in California which is reasonable. 


## 1-2 Genetic Algorithm

Conduct genetic algorithm with classification dataset (wine quality dataset). Compare the result with feature importance based on random forest. 
Hyparparameters are belowed. 

hyperparameter|value
--------------|-----
number of chromosomes| 30
population size| 10      
crossover mechanism:| 0.8 
mutation rate:| 0.15


GA consists with 5 steps 
1. selection 
2. crossover
3. mutation
4. replace previous population with new population
5. evaluate new population and update best chromosome


**selection**
```
    def __selection(
            self, population: np.ndarray, population_scores: np.ndarray,
            best_chromosome_index: int) -> np.ndarray:
        # Create new population
        new_population = [population[best_chromosome_index]]

        # Tournament_k chromosome tournament until fill the numpy array
        while len(new_population) != self.population_size:
            # Generate tournament_k positions randomly
            k_chromosomes = np.random.randint(
                len(population), size=self.tournament_k)
            # Get the best one of these tournament_k chromosomes
            best_of_tournament_index = np.argmax(
                population_scores[k_chromosomes])
            # Append it to the new population
            new_population.append(
                population[k_chromosomes[best_of_tournament_index]])

        return np.array(new_population)
```

**crossover**
```
    def __crossover(self, population: np.ndarray) -> np.ndarray:
        # Define the number of crosses
        n_crosses = int(self.crossover_rate * int(self.population_size / 2))

        # Make a copy from current population
        crossover_population = population.copy()

        # Make n_crosses crosses
        for i in range(0, n_crosses*2, 2):
            cut_index = np.random.randint(1, self.X.shape[1])
            tmp = crossover_population[i, cut_index:].copy()
            crossover_population[i, cut_index:], crossover_population[i+1,
                                                                      cut_index:] = crossover_population[i+1, cut_index:], tmp
            # Avoid null chromosomes
            if not all(crossover_population[i]):
                crossover_population[i] = population[i]
            if not all(crossover_population[i+1]):
                crossover_population[i+1] = population[i+1]

        return crossover_population
```

**mutation**
```

    def __mutation(self, population: np.ndarray) -> np.ndarray:
        # Define number of mutations to do
        n_mutations = int(
            self.mutation_rate * self.population_size * self.X.shape[1])

        # Mutating n_mutations genes
        for _ in range(n_mutations):
            chromosome_index = np.random.randint(0, self.population_size)
            gene_index = np.random.randint(0, self.X.shape[1])
            population[chromosome_index, gene_index] = 0 if \
                population[chromosome_index, gene_index] == 1 else 1

        return population
```

**select best chromosomes**

```
def __update_best_chromosome(
            self, population: np.ndarray, population_scores: np.ndarray,
            best_chromosome_index: int, i_gen: int):
        # Initialize best_chromosome
        if self.best_chromosome[0] is None and self.best_chromosome[1] is None:
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen)
            self.threshold_times_convergence = 5
        # Update if new is better
        elif population_scores[best_chromosome_index] > \
                self.best_chromosome[1]:
            if i_gen >= 17:
                self.threshold_times_convergence = int(np.ceil(0.3 * i_gen))
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen)
            self.n_times_convergence = 0
            print(
                '# (BETTER) A better chromosome than the current one has '
                f'been found ({self.best_chromosome[1]}).')
        # If is smaller than self.threshold count it until convergence
        elif abs(population_scores[best_chromosome_index] -
                 self.best_chromosome[1]) <= self.threshold:
            self.n_times_convergence = self.n_times_convergence + 1
            print(
                f'# Same scoring value found {self.n_times_convergence}'
                f'/{self.threshold_times_convergence} times.')
            if self.n_times_convergence == self.threshold_times_convergence:
                self.convergence = True
        else:
            self.n_times_convergence = 0
            print('# (WORST) No better chromosome than the current one has '
                  f'been found ({population_scores[best_chromosome_index]}).')
        print(f'# Current best chromosome: {self.best_chromosome}')
```


### GA Raw Result 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_GA_results.PNG)

### GA Result 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_GA_wine.PNG)

With genetic algorithm, we found the best variable set [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]. According to the result, the best varialbes for wine quality classification are volatile acidity, residual sugar, chlorides, total sulfur dioxide, pH, alcohol. [1, 3, 4, 6, 8, 10]

### Random Forest Importance Score 

extract importance score from random forest classifier.


```
from sklearn.ensemble import RandomForestClassifier
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('1_2_feature_importance')
```

![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_2_feature_importance.png)

According to above figure, feature 1, feature 6, feature 10 are remarkably important -> volatile acidity, total sulfur dioxide, alcohol. 
Selected variables with GA also have high feature importances in random forest classificaiton. With this result, we can say that GA worked well for selecting variables. 

## 1-3 Principal Component Analysis (PCA) 

Conduct PCA with classification dataset (wine quality dataset) with and without sklearn. Wine quality can be classified to 6 classes (3, 4 ,5 ,6 ,7, 8).


### Explained Variance 


Calculate eigen values and eigen vectors of correlation.

```
## PCA
# without sklearn 
# Standardize the data
X = (X - X.mean()) / X.std(ddof=0)
# Calculating the correlation matrix of the data
X_corr = (1 / 150) * X.T.dot(X)

eig_values, eig_vectors = np.linalg.eig(X_corr)
```

Plotting the variance explained by each principal component

```
explained_variance=(eig_values / np.sum(eig_values))*100
plt.figure(figsize=(8,4))
plt.bar(range(11), explained_variance, alpha=0.6)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')
#plt.show()
```

![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_percentage_of_explained_variance.png)


Percentage of explained variance is decreasing until 5 dimensions and increasing after that. 
The cumulative explained variance is higher than 60% at 4 dimensions. 

### PCA 


Drawing PCA with new principal components pc1 and pc2. 

```
# calculating our new axis
pc1 = X.dot(eig_vectors[:,0])
pc2 = X.dot(eig_vectors[:,1])

def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    unique = list(set(y))
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    
    for i, spec in enumerate(y):
        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=colors[unique.index(spec)])
        ax.annotate(str(i+1), (pc1[i],pc2[i]))
    
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)
    
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
    
    plt.clf()
    plt.grid()
    plt.axis([-4, 4, -3, 3])
    plt.show()
 ```
 
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_pca_results.png)

Plotting PCA without sklearn.It is hard to find some features by classes. However, class 5, which marked as green, tend to be concentrated in the lower middle. 


### PCA with sklearn  

![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_3_pca_sklearn.png)

Plotting PCA with sklearn. Since it is also hard to find features, we can say this data (wine quality dataset) has few characteristics between groups. In addition, PCA without sklearn is well designed because the result is quite close to the above one. 


## 1-4 Locally Linear Embedding (LLE) 

Conduct LLE wth classification dataset (wine quality dataset) with various n_neighbors. The experiment was conducted by changing number of neighbors to 2, 3, 5, and 10. LLE maintains locality information unlike MDS, but it does not care about other things. Usually, LLE is sensitive with number of neighbors, but it is not in this dataset. 

### LLE
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results2.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results3.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results5.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results10.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_LLE_results20.png)

At every case, LLE seems almost same. We could not find the difference between various n_neighbors. 

## 1-4 t-distributed Stochastic Neighbor Embedding (t-SNE)  

Conduct t-SNE wth classification dataset (wine quality dataset) with various perplexity. Perplexity in t-SNE is a parameter controlling a trade-off between local and global structure of data. Perplexity is defined as 2^entropy and it has a uniform distribution. Thus, if data has high entropy, perplexity is also high. Lower perplexity considers more to local information. In contrast, higher perplexity puts more weights to global information, so it makes the probability values that all points can have similarities. 

### t-SNE
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE2.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE3.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE5.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE10.png)
![](https://github.com/goeunchae/Business-Analytics_1/blob/main/pics/1_4_t-SNE20.png)

We can capture the difference between various perplexity. When perplexity is in [2, 3 ,5], all of them are similar and they take the shape of a circle. Then, as perplexity increases, it generally changes to a shape of V. However, the optimal perplexity does not exist and also there is no meaning of the distance among clusters in t-SNE.At every pereplexity, no speical clusters were formed. It can be seen that the characteristics of each wine quality class are not clear. 
