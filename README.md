 Naive Bayes Classification on Coursera Course Reviews

The project focuses on implementing a Naive Bayes Classifier to analyze and classify Coursera course reviews based on their ratings, ranging from 1 to 5. The dataset used consists of 107,013 reviews. The goal was to preprocess the data, train the model, and evaluate its performance using various metrics.


1. Dataset Information
   
    •Total Samples: 140,000

    •Training Set: 85,610 samples (80%)

    •Test Set: 21,403 samples (20%)

    •Labels: 5 classes (1 to 5)

    •Observation: The dataset exhibited a significant bias, with 73% of reviews labeled as 5. This class imbalance posed a challenge for the model’s performance.



2. Data Preprocessing

    •Removing non-English reviews
    
    •Removing stop words
    
    •Converting text to lowercase
    
    •Eliminating HTML tags and punctuation
    
    •Lemmatization: Implemented using custom functions like remove_html_tags, remove_punctuation, and lemmatize_text.



3. Approach and Implementation

    •Data Loading and Preprocessing: Applied functions to clean and prepare data.
   
    •Data Splitting: Used train_test_split to divide data.
   
    •Tokenization: Applied word_tokenize from nltk.
   
    •Prior Probability Calculation: Custom code was written.
   
    •Token Probability with Laplace Smoothing: Enhanced model accuracy.
   
    •Classification: Custom function classify_document predicted test labels.
   
    •Evaluation Metrics: Utilized confusion matrix and other metrics to analyze performance.
   
    •User Input Classification: Custom function enabled real-time review classification.



4. Evaluation and Results
    •Confusion Matrix: Created for different training set sizes (80% and 70%).
   
    •Metric Evaluation:
   
      The model's accuracy was influenced by the class imbalance, skewing results towards label 5.
   
      Precision and recall metrics highlighted this bias, necessitating further adjustments for a balanced evaluation.



5. Challenges and Improvements

Challenges:
    
  •Managing class imbalance.
  
  •Handling diverse data types, including non-English text and special characters.
  
  •Ensuring code efficiency and scalability.

Improvement Suggestions:
 
  •Implement SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
  
  •Experiment with alternative algorithms like SVM or Random Forest for comparative analysis.
  
  •Optimize code and enhance error handling.



6. Conclusion
   
The project successfully demonstrated the implementation of a Naive Bayes Classifier for sentiment analysis of Coursera reviews. Despite challenges such as class imbalance and preprocessing complexities, the model provided valuable insights. Future work could focus on addressing these limitations to improve accuracy and reliability.







