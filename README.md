
---
# GoEmotions Text Classifier

This project is a multi-class text classifier built on the GoEmotions dataset. The dataset consists of 58k comments extracted from Reddit, human-annotated into 27 emotion categories or "neutral".

## Overview

The task involves building a classifier prototype capable of predicting emotion categories based on the text from the Reddit comments. The project takes an innovative approach by selecting only a subset of the labels for focused classification, while the remaining labels are merged into a new category.

1. **Objective**: The project aims to process and classify text data, uncovering hidden topics and patterns within a dataset.

2. **Data Preprocessing**:
   - **Text Vectorization**: The text data is converted into a numeric format using `TfidfVectorizer`, which emphasizes unique terms in the documents.
   - **Stop-word Removal**: Common words that do not contribute much to the meaning of the text are removed to create a cleaner dataset.
   - **Train-Test Split**: The dataset is divided into training and testing sets to evaluate the model's performance accurately.

3. **Modeling**:
   - **Topic Modeling**: Techniques like LSA, LDA, and NMF are used to extract topics from the text data.
   - **Classification**: Logistic regression and k-Nearest Neighbors (kNN) classifiers are trained to categorize the text data. A 1-dimensional Convolutional Neural Network (1D CNN) is also mentioned for text classification.

4. **Evaluation**:
   - **Performance Metrics**: Models are evaluated using accuracy and F1 scores, with moderate overall performance reported.
   - **Classification Reports**: Detailed reports provide insights into the precision, recall, and F1 scores for different classes.
   - **Confusion Matrix**: The ability of the models to correctly classify different emotion classes is visualized.

5. **Results**:
   - **Topic Weights**: A box plot visualization shows the distribution of topic weights, indicating the importance and prevalence of each topic in the dataset.
   - **Accuracy**: The models achieve varying levels of accuracy, with some models reaching an accuracy of around 65%.

6. **Insights**:
   - The project provides insights into the importance of preprocessing and the impact of different techniques on model performance.
   - The topic modeling uncovers the underlying structure of the dataset, which can be used to improve classification performance.

7. **Challenges**:
   - It is noted that some topics might not correspond directly to the labels in the dataset since LDA is an unsupervised method.

## Data

The data used for this project is the GoEmotions dataset, comprising 58,000 carefully curated comments extracted from Reddit, labeled into 27 emotion categories and a "neutral" category.

## Dependencies

List the libraries and tools used in the project. For example:
* Python 3.x
* NumPy
* Pandas
* TensorFlow/Keras
* NLTK
* etc.

## Installation & Usage

Instructions on how to install and run your code. For example:

```sh
git clone https://github.com/kashyapHebbar/NaturalLanguage.git
cd go_emotions_serving 2
```


## Methodology

A brief overview of your method. For example, if you used a specific machine learning model, preprocessing steps, feature extraction, etc. 

## Results

Summarize the results of your project. What kind of accuracy did you achieve? How well did your model perform? Add graphs or other visual aids if possible.

## Future Improvements

Outline any future improvements that could be made to this project, such as other models that could be tested or other data that could be used.

## Acknowledgements

This project uses the [GoEmotions dataset](link_to_dataset), which is publicly available.

---


