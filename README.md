
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

**Data Preprocessing and Feature Extraction:**
1. **Text Length Distribution**: The methodology included calculating the text length of each sample and visualizing the distribution with a histogram, which informs the choice of preprocessing techniques or model architecture.

**Natural Language Processing Algorithms:**
2. **CNN**: A 1D Convolutional Neural Network model was implemented for NLP tasks using GloVe word embeddings for feature representation. The methodology involved loading embeddings, splitting data, tokenizing, padding sequences, encoding labels, and defining the CNN model with TensorFlow’s Keras API.

3. **Random Forest Classifier**: The methodology included using TfidfVectorizer for text data vectorization, splitting data into training and testing sets, training the classifier, and evaluating it using K-Fold cross-validation.

4. **Latent Dirichlet Allocation (LDA)**: The approach used CountVectorizer for text data conversion into a bag-of-words representation, topic extraction using LDA, and visualization of the distribution of topics.

5. **LSA (Latent Semantic Analysis)**: The methodology involved using TfidfVectorizer, splitting data, applying LSA for topic modeling, fitting the LSA model, and visualizing topic weights.

**Preprocessing Techniques:**
6. **Removing Stop-words**: The text data was pre-processed by removing stopwords, using TfidfVectorizer with an additional stop-words parameter, splitting the dataset, training a logistic regression model, and evaluating the model’s performance.

**Discussion of Best Results:**
7. **Data Preprocessing**: Different preprocessing techniques were experimented with, such as lowercasing, removing stopwords, stemming, lemmatization, and removing special characters. Lowercasing yielded the best results.

8. **Text Featurization**: Various text featurization methods were tested, including Bag Of Words, TF-IDF Vectorizer, Word Embeddings, N-grams, Bi-Grams, Tri-Grams, LDA, NMF, and LSA. The TF-IDF method showed the best performance.

## Results

**Data Preprocessing and Feature Extraction:**
- **Lowercasing** as a preprocessing technique yielded the best results with an accuracy of 0.66 and a weighted F1 score of 0.65.
- **TF-IDF Vectorization** method showed the best performance in text featurization with an accuracy of 0.65 and a weighted F1 score of 0.65.

**Topic Modeling:**
- **LSA**: Identified five topics with topic weights ranging from -0.6 to 0.8, indicating a mix of positive and negative weights for topics.
- **NMF**: Revealed that Topic 4 had the highest median weight, suggesting higher importance in the dataset.

**Word Embeddings:**
- Utilized **pre-trained GloVe embeddings** to represent text, which is expected to capture semantic meaning effectively.
- The logistic regression model trained with GloVe embeddings showed generally higher correct predictions on the confusion matrix.

**Sentiment Analysis:**
- The histogram of sentiment polarity scores ranged from -1 (most negative) to 1 (most positive), offering insights into the overall sentiment distribution within the dataset.

**Model Performance:**
- **Legacy_RMSprop** optimizer achieved the best test accuracy (0.524978) and test F1 score (0.486597).
- **Adadelta** optimizer was noted for addressing the aggressive decrease in learning rates from Adagrad, resulting in a more robust optimization method.

**Overall Outcome Evaluation:**
- The project built emotion classification models using a variety of techniques and algorithms, including deep learning-based algorithms (GRU, Bi-Directional LSTM, CNN) and machine learning algorithms (kNN, XGBoost, Random Forest).

**Discussion of Best Results:**
- The best preprocessing results came from lowercasing, with an accuracy of 0.66 and a weighted F1 score of 0.65.
- The best text featurization results were obtained using the TF-IDF method, with an accuracy of 0.65 and a weighted F1 score of 0.65.
- The 3-grams model had the lowest accuracy at 0.40.

**Additional Observations:**
- The top 20 most frequent words in the text samples were identified, providing insights into the predominant topics and emotions present in the text samples.
  ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/e17ce69d-3448-45be-9040-4c19662f5745)

- The distribution of text lengths in the training dataset was visualized, which can inform the choice of preprocessing techniques or model architecture.

- **Convolutional Neural Network (CNN)**: The CNN models showed varying results. One implementation achieved a test accuracy of approximately 59.68%, indicating a moderate level of performance. Another CNN model, which used Binary Focal Loss and pre-trained Word2Vec embeddings, had a lower test accuracy of 29.38%, suggesting it struggled to generalize to new data. The document suggests that the CNN models may need further refinement in terms of architecture, hyperparameters, or preprocessing to achieve better accuracy.
  ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/bc8c5e93-c5f1-4d9a-b2a3-43af0a15aa20)


- **Random Forest**: The Random Forest Classifier, which used TfidfVectorizer for text data vectorization and was evaluated using K-Fold cross-validation, did not perform as well, with a test accuracy of 29%. This indicates that the model may not have been complex enough to capture the patterns in the data or that the data itself was challenging for this type of model.
  ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/c09eb203-168e-49c6-9d09-cb23635d3c2f)


- **GRU (Gated Recurrent Unit)**: The GRU model achieved a test accuracy of approximately 58%, which is a reasonable performance, especially considering that GRU is adept at handling sequential data and can capture temporal dependencies.
  ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/b504c7ff-5dd2-4db2-9ad6-241d103da5dc)


- **k-Nearest Neighbors (kNN)**: The kNN model achieved an accuracy of 50%, which is on the lower end compared to deep learning models. This might be due to the high dimensionality of the data or the simplicity of the kNN algorithm, which relies on distance metrics for classification.
   ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/bcd713bc-b73a-4914-b60c-e5b174bae1bd)


- **XGBoost**: The document does not provide a specific accuracy for the XGBoost model, but as an optimized gradient boosting algorithm, it's typically known for its performance in various machine learning tasks.
   ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/9f63ddff-4b06-449a-a786-6a33c3a29195)
  
- **Bi-Directional LSTM**: The Bi-Directional LSTM, which utilizes two LSTM layers to process data from both past and future states, is expected to perform well on sequential data like text gives the accuracy as 65 % .
  ![image](https://github.com/kashyapHebbar/NaturalLanguage/assets/65105317/00957567-7351-4228-acbd-c2dfdafb900c)


In summary, the deep learning models, particularly the GRU, showed promising results, while traditional machine learning models like Random Forest and kNN displayed lower accuracy. The CNN's performance varied significantly depending on the specific implementation and choice of loss function. The document suggests that all models could potentially benefit from hyperparameter tuning and further refinement of their architectures.



This summary encapsulates the key findings and outcomes of the experiments and analyses conducted in the project. The results indicate a nuanced understanding of the data and the performance of various models and techniques.

## Future Improvements

1. **Model Refinement**: For models that did not perform as expected, consider hyperparameter tuning, adding additional layers, or employing more complex architectures. Addressing class imbalance with a class-weighted loss function could also be beneficial.

2. **Efficiency vs. Quality**: For well-performing models, there's a possibility to increase efficiency by reducing complexity or adjusting learning parameters. This might involve a trade-off with accuracy, which should be carefully considered based on the application context.

3. **Extended Training**: Some models might benefit from more training epochs or hyperparameter tuning to achieve better accuracy. Monitoring the gap between training and validation accuracy can help in identifying overfitting.

4. **Feature Extraction and Preprocessing**: Further exploration of preprocessing techniques and feature extraction methods could lead to improvements. For instance, examining the impact of different featurization methods like TF-IDF, Bag of Words, and word embeddings on model performance.

5. **Evaluation Metrics**: It's crucial to consider multiple evaluation metrics, such as accuracy, precision, recall, and F1 score, to gain a comprehensive understanding of a model's effectiveness.

6. **Topic Modeling**: Refining topic modeling techniques like NMF and LSA could uncover more nuanced themes or patterns in the text data, which can be useful for improving the underlying structure of the dataset.

7. **Semantic Meaning**: Bigram analysis and other n-gram techniques could be expanded to capture more semantic meaning in the text, which might improve the performance of emotion classification models.

8. **Class Differentiation**: Investigating the linguistic characteristics of each emotion could reveal potential features for differentiation, which may help in improving the classification model.

9. **Optimization Methods**: Evaluating different optimization methods and their parameters could lead to better model performance, as indicated by the success of Legacy_RMSprop in the document.

10. **Class Merging**: The strategy of merging related emotion categories into broader classes could be refined to ensure that the model captures the nuances between different emotions effectively.

These suggestions aim to build on the current work and guide future research and development efforts to enhance the performance and efficiency of the emotion classification models discussed in the document.

## Acknowledgements

This project uses the [GoEmotions dataset](link_to_dataset), which is publicly available.

---


