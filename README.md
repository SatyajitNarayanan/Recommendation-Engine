# Recommendation-Engine
Created a recommendation engine as a part of the Python Programming course at NC State

## Problem:
To create a recommendation engine that predicts the likeliness of a user liking a movie based on past user behavior and data on movies

### Data Used:
* The MovieLens dataset.
* Collected by the GroupLens Research Project at the University of Minnesota. It consists of:
** 100,000 ratings (1-5) from 943 users on 1682 movies
** Demographic info for the users
** Genre information of movies

## Approach:
There are broadly two types of recommender engines – Content Based and Collaborative Filtering
* Content Based algorithms 
* Collaborative Filtering algorithm: Further, there are 2 types of collaborative filtering algorithms:
  * User-User Collaborative filtering 
  * Item-Item Collaborative filtering

* Predictions are made using Collaborative Filtering method for both:
  * Existing Users
  * New User

## Features incorporated

Hybrid prediction score
* (1- α) x User Prediction + α x Movie Prediction
