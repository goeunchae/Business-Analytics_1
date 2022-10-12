import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings(action='ignore')

df = pd.read_csv('./data/data_housing.csv')

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

selected_variables, steps, adjusted_r_squared, sv_per_step = FC(df)

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
    
result_pic(selected_variables, steps, adjusted_r_squared, sv_per_step)

# Backward Elimination 
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


selected_variables, steps, adjusted_r_squared, sv_per_step = BE(df)
    
result_pic(selected_variables, steps, adjusted_r_squared, sv_per_step)

# Stepwise Selection 
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

selected_variables, steps, adjusted_r_squared, sv_per_step = SS(df)
result_pic(selected_variables, steps, adjusted_r_squared, sv_per_step)

