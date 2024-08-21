# Fake News Detection Using Machine Learning

## Overview

Imagine this project as a superhero fighting against fake news, the villain of our digital world. Using fancy technology called machine learning, it's like giving our superhero superpowers to spot fake news articles.

In today's world, **fake news** is like a big problem we all face. It's like a bunch of lies mixed with truth, and it's hard to tell them apart. So, this project is like our hero, working hard behind the scenes to help us know what's real and what's fake.

Inside this project, there's a lot of cool stuff going on. I'm talking about sorting through **tons of information**, training our superhero to be really good at spotting fake news, and then giving it a simple tool to help us out.

So, come join me on this adventure as we fight against fake news together. With your support, we can make sure the truth always wins!

## Features

- **Data Loading**: Load the dataset from a CSV file (`train.csv`) using pandas.
- **Data Preprocessing**: Clean and preprocess the text data by combining author and title, removing non-alphabetic characters, converting to lowercase, and stemming words.
- **Feature Engineering**: Utilize TF-IDF (Term Frequency-Inverse Document Frequency) to transform text data into numerical features suitable for machine learning models.
- **Model Training**: Train a logistic regression model on the preprocessed data to classify news articles as real or fake.
- **Model Evaluation**: Evaluate the trained model's performance on both the training and testing datasets using accuracy score.
- **Prediction System**: Implement a simple prediction system to classify news articles as real or fake based on input text data.


### Prediction System

To classify news articles as real or fake using the trained model, run the `predict.py` script. You can provide input data as text or use the provided test data.








