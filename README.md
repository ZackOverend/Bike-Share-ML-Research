# Abstract
This project utilizes supervised machine-learning (SVP) regression to predict the daily bike rental counts and analyze important counts using the UCI Bike Sharing Dataset. After cleaning the categories and categorizing discrete features, the models were trained and compared using PyCaret. The models tested included CatBoost, Extra Trees, Random Forest, Linear Regression and many others. The best overall performance was achieved by the CatBoost Regressor, which produced strong performance with a lower RMSE, MAE, and MAPE. However, Extra Trees also performed competitively with a lower MAE and MAPE. Final evaluation demonstrated that overall the work-related variables were most influential for this dataset, followed by the weather. Most influentially, whether it was a working day yielded more bike usage, followed closely by the “feels like” temperature outside. This makes sense in terms of biking in a highly populated area, where in cold weather commuters may opt to use other forms of transit. Overall, the project shows how modern ML tools can benchmark regression models and reflect the expected real-world demand for bike sharing systems.

# Video Demonstration
https://youtu.be/XBpmZ_sVYl4

# Dataset
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
