
# Concrete Compressive Strength Prediction

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period.
I solved this problem using Data science and Machine learning technology, developed a web application which predicts the "Concrete compressive strength" based on the quantities of raw material, given as an input. Sounds like this saves a lot of time and effort right !

`Data source:-` https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set

## Approach:
1. Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.
2. Performed Exploratory data analysis:
    - First viewed the distribution of the target feature, "Concrete compressive strength", which was in Normal distribution with a very little right skewness.
    - Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
    - To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
    - Checked for the presence of outliers in all the columns and found that the column 'age' is having more no. of outliers. Removed outliers using IQR technique, in which I considered both including and excluding the lower and upper limits into two separate dataframes and merged both into a single dataframe. This has increased the data size so that a Machine learning model can be trained efficiently. 
3. Experimenting with various ML algorithms:
    - First, tried with Linear regression models and feature selection using Backward elimination, RFE and the LassoCV approaches. Stored the important features found by each model into "relevant_features_by_models.csv" file into the "results" directory. Performance metrics are calculated for all the three approaches and recorded in the "Performance of algorithms.csv" file in the "results" directory. Even though all the three approaches delivered similar performance, I chose RFE approach, as the test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression. But the disadvantage is, model showed slight underfitting.
    - Next, tried with various tree based models, performed hyper parameter tuning using the Randomized SearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model, recorded that info into a "relevant_features_by_models.csv" file into the "results" directory. Built models, evaluated on both the training and testing data and recorded the performance metrics in the "Performance of algorithms.csv" file in the "results" directory.
    - Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor. Saved these two models into the "models" directory.
4. Deployment:
    Deployed the XGBoost regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML5.

At each step in both development and deployment parts, logging operation is performed which are stored in the development_logs.log and deployment_logs.log files respectively. 

So, now we can find the Concrete compressive strength quickly by just passing the quantities of the raw materials as an input to the web application 😊. 


## Web Deployment
Deployed on web using Heroku (PaaS) 
url:- https://ccs-predictor.herokuapp.com/
## Screenshots

![image](https://user-images.githubusercontent.com/22242325/134782546-83e740aa-c0c5-45d9-a302-6f30b4362d7e.png)

## Tools and technologies used

![tools and technologies used](https://user-images.githubusercontent.com/22242325/135045139-7e9e2bdd-ae1a-43f7-9c8b-5d12e4a4cd62.png)

## High level design
URL:- https://drive.google.com/file/d/1ucWKT8LXsr2k-9X83UA5YjB7GSRxY8ND/view?usp=sharing

## Low level design
URL:- https://drive.google.com/file/d/1ZztcOqvvFe7Xhjhhljnpu4OqfbQvj6iW/view?usp=sharing

## Architecture
URL:- https://drive.google.com/file/d/1IfsVoRv1WqYB4qZYLT9LBTiUF1HLZ3q_/view?usp=sharing

## Detailed project report
URL:- https://drive.google.com/file/d/1LyuLnu5ZlIX-V6vVvLUggRfnnskqufZ5/view?usp=sharing

## Wireframe document
URL:- https://drive.google.com/file/d/1XmkH9p1uVLYUY-y6-mSRiJZhf-WhXuJ1/view?usp=sharing

## Demo video
URL:- https://youtu.be/k0VswDZ1waA

  
## References
 
 - [Testing the compressive strength of a concrete in laboratory](https://www.youtube.com/watch?v=t4RDdn6rOwU&ab_channel=Anime_Edu-CivilEngineeringVideos) 
 - [Concrete Basics: Essential Ingredients For A Concrete Mixture](https://concretesupplyco.com/concrete-basics/)
 - [Applications of Fly ash](https://www.thespruce.com/fly-ash-applications-844761)
 - [Blast furnace slag cement](https://theconstructor.org/concrete/blast-furnace-slag-cement/23534/)
 - [Applications of Superplasitcizer in concrete making](https://en.wikipedia.org/wiki/Superplasticizer)
 - [Factors that affect strength of concrete](https://gharpedia.com/blog/factors-that-affect-strength-of-concrete/)
 - [Feature selection with sklearn and pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
 - [sklearn's LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
 - [Post pruning technique in Decision tree algorithm ](https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09)
 - [Hyper parameter tuning in XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
 - [HTML, CSS tutorials ](https://www.w3schools.com/)
## Author

- Sesha Venkata Sriram Erramilli [LinkedIn](https://www.linkedin.com/in/svsriramerramilli/)

  
