# Amazon Bestseller Classification
Captured the feature data (12 items in total, such as sales, evaluation, etc.) of thousands of products on Amazon, improved the accuracy of the classification of best-selling products by analyzing with 8 kinds of algorithms (K- k nearest neighbor, support vector machine, decision tree, random forest, gradient boosted decision tree, random forest, Gaussian NB, Logistic Regression).

These data files are after clean data files, not orignal files.

## 1. Scrapy the data from amazon web pages
Captured feature data, wrote data handlers in Python, and label them 0 and 1 as not bestseller data and bestseller data.

## 2. Clean the data to numeric data for modeling
Give up the name feature cause it not affects the final result

## 3. Classifiy those products' features and improve the accuracy of algorithms
Improved the accuracy by grid search, and displayed the comparison result with confusion matrix.
