# Anime Recommendation System
This project is made for a data competition called TSDC (Tamkang Statistics Data Club). This project focuses on building an anime recommendation system by utilizing various machine learning models. With the rise of information overload in today’s digital age, recommendation systems play a critical role in helping users discover content that is relevant and engaging to them. This research aims to explore and implement models capable of accurately predicting user preferences based on historical anime ratings.

## Project Motivation:

Recommendation systems are widely used by platforms such as e-commerce and streaming services to enhance user experience and drive engagement. Inspired by the need for personalized content recommendations, this project leverages a dataset provided by Kaggle containing user ratings for various anime shows. The goal is to predict how users will rate anime they haven’t seen, based on their past behavior, and recommend high-rated content.

## Methods:

### Three machine learning models were used to build the recommendation system:

 1.	K-Nearest Neighbors (KNN) - A collaborative filtering method that finds similarities between users or items
 2.	Funk Singular Value Decomposition (SVD) - A matrix factorization technique that handles sparse data efficiently.
 3.	Factorization Machines (FM) - A model that analyzes user-item interactions to make more accurate recommendations.

Each model was evaluated using Root Mean Square Error (RMSE), and KNN was identified as the optimal model due to its balance of accuracy and computational efficiency.

### The data of this project is Anime Recommendations Database provided by CooperUnion in Kaggle. 
