# N-Model-XGBoost-Ensemble-w-Bayesian-Optimization
Example code for how to produce an N-model XGBoost ensemble model by changing the seed for the train/CV split and also using Bayesian Optimization for hyperparameter tuning.

The example code is made to run with a housing prices data set available on Kaggle here: 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/Data

There is some data cleaning in the script and then the start of the model building begins with the global variables.  The Bayesian optimization for hyperparameter tuning can be done using a single xgboost model using the function xgb_eval_single or multiple models can be used by using the function xgb_eval_multi.  The number of models is set as a global variable.  The final model is built using the run_single or multi_model functions depending on if you want one or more models.
