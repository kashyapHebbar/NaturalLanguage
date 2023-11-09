
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

1. **Text Data**: The dataset consists of textual data that requires processing and vectorization before it can be used for machine learning tasks.

2. **Preprocessing Steps**:
   - **Vectorization**: The `TfidfVectorizer` is used to transform the text data into a TF-IDF representation, which suggests that the data includes textual features that need to be quantified for analysis.
   - **Stop-word Removal**: The preprocessing includes the removal of stopwords, indicating that the dataset contains common English words that are not relevant to the analysis.

3. **Splitting**: The data is split into training and testing sets, with an 80-20 ratio mentioned in one of the sections, which is a common practice in machine learning to validate the model's performance on unseen data.

4. **Labeling**: There is a mention of emotion classes and a classification report, which implies that the text data is labeled, possibly for a sentiment analysis or emotion detection task.

5. **Model Training**: The data is used to train various models, including logistic regression, kNN, and a 1D CNN, which suggests that the dataset is suitable for supervised learning tasks.

6. **Topic Modeling**: The use of LSA, LDA, and NMF for topic modeling indicates that the dataset is rich enough to extract multiple topics, which could mean that the text comes from a diverse set of documents or a large corpus with varied content.

7. **Performance Evaluation**: The dataset's complexity and the challenge of the task are hinted at by the moderate performance metrics (accuracy and F1 scores) achieved by the models.

8. **Emotion Classes**: The dataset seems to have been categorized into different emotion classes, as indicated by the performance evaluation on classes like ‘Neutral’, ‘Empathy’, ‘Elation’, ‘Desire’, ‘Apprehension’, and ‘Confusion’.


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


1. **Preprocessing**:
   - **Text Vectorization**: The text data is transformed into a numerical format using `TfidfVectorizer`, which suggests that Term Frequency-Inverse Document Frequency (TF-IDF) is used to give more weight to unique words in the documents.
   - **Stop-word Removal**: Common words that are typically considered noise in text data (stop-words) are removed to focus on more meaningful words.
   - **Lemmatization**: This step is implied to standardize words to their base or root form, which helps in reducing the complexity of the text data.

2. **Feature Extraction**:
   - **TF-IDF Representation**: This is used to convert text data into a matrix of TF-IDF features, which reflects how important a word is to a document in a collection or corpus.
   - **Word Embeddings**: There is a mention of using Word2Vec to create word embeddings, which suggests that the project also explores dense vector representations of words that capture the context of a word in a document.

3. **Modeling**:
   
i. **Topic Modeling Techniques**:
   - **Latent Semantic Analysis (LSA)**: This technique is used for extracting and representing the contextual-usage meaning of words by statistical computations applied to a large corpus of text. LSA is based on singular value decomposition (SVD) which reduces the dimensionality of the TF-IDF matrix, capturing the underlying structure in the data.
   - **Latent Dirichlet Allocation (LDA)**: LDA is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. It's particularly used for identifying topics in a set of documents, assuming that each document is a mixture of a small number of topics.
   - **Non-negative Matrix Factorization (NMF)**: NMF is a group of algorithms in multivariate analysis where a matrix V is factorized into (usually) two matrices W and H, with the property that all three matrices have no negative elements. This non-negativity makes the resulting matrices easier to inspect.

ii. **Classification Models**:
   - **Logistic Regression**: A statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In the context of text classification, logistic regression can be used to predict the probability that a given text belongs to a certain category.
   - **k-Nearest Neighbors (kNN)**: A non-parametric method used for classification and regression. In kNN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors.
   - **1-Dimensional Convolutional Neural Network (1D CNN)**: A type of neural network that is particularly well-suited for processing sequences of data. For text, 1D CNNs can capture the spatial hierarchy in data by applying convolutional layers to the sequence, allowing the model to detect complex patterns such as phrases or sentences.

4. **Model Evaluation**:
   - **Accuracy and F1 Score**: These metrics are used to evaluate the overall performance of the models.
   - **Classification Report**: Provides detailed performance metrics for each class, including precision, recall, and F1 score.
   - **Confusion Matrix**: Visualizes the performance of the classification model, showing the correct and incorrect predictions across different classes.

5. **Hyperparameter Tuning**:
   - **Optimization**: For kNN, the optimal number of neighbors is determined by iterating through a range of values and evaluating the accuracy.
   - **Fine-tuning**: The LDA model is fine-tuned by adjusting hyperparameters to better capture the underlying topics in the dataset.

6. **Visualization**:
   - **Box Plots**: Used to visualize the distribution of topic weights for the topic modeling methods, which helps in understanding the importance of each topic in the dataset.
   - **Word Importance**: For LDA, the top words for each topic are displayed, providing insights into the prevalent themes.

The methodology combines traditional machine learning techniques with more advanced models and a variety of preprocessing and feature extraction methods to analyze text data effectively. The approach is systematic and iterative, with an emphasis on both model performance and the interpretability of results.

## Results

Summarize the results of your project. What kind of accuracy did you achieve? How well did your model perform? Add graphs or other visual aids if possible.

## Future Improvements

Outline any future improvements that could be made to this project, such as other models that could be tested or other data that could be used.

## Acknowledgements

This project uses the [GoEmotions dataset](link_to_dataset), which is publicly available.

---


