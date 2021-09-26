
# Concrete Compressive Strength Prediction

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period.
I solved this problem using Data science and Machine learning and developed a web application which predicts the "Concrete compressive strength" based on the quantities of raw material, given as an input. Sounds like this saves a lot of time and effort right !😀

`Data source:-` https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set

## Approach:
1. Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.
2. Performed Exploratory data analysis:
    - First viewed the distribution of the target feature, "Concrete compressive strength", which was in Normal distribution with a very little right skewness.
    - Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
    - To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
    - Checked for the presence of outliers in all the columns and found that the column 'age' is having more no. of outliers. Removed outliers using IQR technique, in which I considered both including and excluding the lower and upper limits into two separate dataframes and merged both into a single dataframe. This has increased the data size so that a Machine learning model can be trained efficiently. 
3. Experimenting with various ML algorithms:
    - First, tried with Linear regression models and feature selection using Backward elimination, RFE and the LassoCV approaches. Stored the important features found by each approach into a "relevant_features_by_models.csv" file into the "results" directory. Performance metrics are calculated for all the three approaches and recorded in the "Performance of algorithms.csv" file in the "results" directory. Even though all the three approaches delivered similar performance, I chose RFE approach as the test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression. But the disadvantage is, model showed slight underfitting.
    - Next, tried with various tree based models, performed hyper parameter tuning using the Randomized SearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model, recorded that info into a "relevant_features_by_models.csv" file into the "results" directory. Built models, evaluated on both the training and testing data and recorded the performance metrics in the "Performance of algorithms.csv" file in the "results" directory.
    - Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor. Saved these two models into the "models" directory.
4. Deployment:
    Deployed the model using Flask as a backend part while for the frontend Web page, used HTML and CSS.

At each step in both development and deployment parts, logging operation is performed which are stored in the development_logs.log and deployment_logs.log files respectively. 

So, now we can find the Concrete compressive strength quickly by just passing the quantities of the raw materials as an input to the web application 😊. 


## Deployment

To deploy this project run

```bash
  npm run app.py
```

```bash
  pip install -r requirements.txt
```
## Screenshots

![image](https://user-images.githubusercontent.com/22242325/134782546-83e740aa-c0c5-45d9-a302-6f30b4362d7e.png)

## Tools and technologies used

![Logo](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.anaconda.com%2F&psig=AOvVaw1HGYfTuNgPvMmEf_dmxD4g&ust=1632723964685000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCPjsjfGAnPMCFQAAAAAdAAAAABAD)
  
## References

 - [Concrete Basics: Essential Ingredients For A Concrete Mixture]( https://concretesupplyco.com/concrete-basics/)
 - [Feature selection with sklearn and pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
 - [sklearn LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
 - [Post pruning technique in Decision tree algorithm ](https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09)
 - [Hyper parameter tuning in XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
 - [Html, CSS tutorials ](https://www.w3schools.com/)
## Authors

- [@esvs2202](https://github.com/esvs2202)

  
