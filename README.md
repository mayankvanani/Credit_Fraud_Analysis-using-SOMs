# Credit_Fraud_Detection-using-SOMs

## Overview
**Self Organising Maps (SOMs)** are very efficient way of analyzing huge dataset as it reduces the dimentionality of the dataset to 1 or 2 dimentions (at max) in order to make visualisation easier. These belong to Unsupervised Models. Since SOMs are not available in any library, **minisom.py** - a prebuilt code containing readily usable _CLASS: Minisom_ is used to implement the SOM.

**Credit_Fraud_Detection.py** - Deals with imlementing the SOM to build **Frequency Map** in order to identify the outliers.

**Credit_Fraud_Prediction.py** - Deals with implementing teh Detection Technique using SOMS (Same as above) and **using ANN to build a classifier that predicts the probability of fraudulent activity by the customers**.

## Dataset 
The Dataset taken from **UCI machine learning repository called _"Statlog (Australian Credit Approval) Dataset"_.** It contains various credentials of 690 customers of a Bank. Last column contains whether the application given by that particular customers was approved (1) or not (0). We need to detect the fraudlent activity.

_NOTE: The credentials ofthe customers have been encoded._

## FRAUD DETECTION

## FRAUD PREDICTION
