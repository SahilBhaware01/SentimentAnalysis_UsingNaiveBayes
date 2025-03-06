import numpy as np
import pandas as pd
import sys
import re
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.model_selection import train_test_split
import time
import nltk
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================================================================>>>Command Line Arguments
sys.stderr = open('/dev/null', 'w')
if len(sys.argv) != 2:
    print("Please provide the correct number of input arguments.")
    sys.exit(1)
try:
    training_datasize = float(sys.argv[1])
except ValueError:
    print("Training size must be a number")
    sys.exit(1)

if not (20 <= training_datasize <= 80):
    print("Training size must be between 20 and 80.")
    sys.exit(1)

# ===============================================================================>>> Load Data
print("Bhaware Sahil A20552865 solution:")
print("Training set size:",training_datasize,"%")



dataset=pd.read_csv("Couresa Ratings/reviews.csv")
# print("All reviews:",len(dataset))
num_rows = int(len(dataset)*0.05)
df_subset = dataset.iloc[:num_rows].copy()
# print("All reviews:",len(df_subset))

# ===============================================================================>>> html and punctuation

def remove_html_tags(text):
    remove_html = re.sub(r'<.*?>', '', text)
    return remove_html

def remove_punctuation(text):
    remove_punctuation_marks = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return remove_punctuation_marks

df_subset['Review'] = df_subset['Review'].apply(remove_html_tags)
# print("HTML tags removed..")

df_subset['Review'] = df_subset['Review'].apply(remove_punctuation)
# print("Punctuation removed..")

# ===============================================================================>>> Remove non english charecters

def is_strictly_english(text):
    pattern = re.compile("^[a-zA-Z0-9 .,;:'\"!?()-]+$")
    return bool(pattern.match(text))

English_reviews = df_subset[df_subset['Review'].apply(is_strictly_english)].copy()
# print("English rows:",len(English_reviews))

# ===============================================================================>>>  Lowercase 

English_reviews['Review'] = English_reviews['Review'].str.lower()
# print("Reviews lowercased..")

# ===============================================================================>>>  Remove stopwords

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    return " ".join(filtered_tokens)

# English_reviews.to_csv("reviews_before_stopwords_removal.csv", index=False)
English_reviews['Review'] = English_reviews['Review'].apply(remove_stopwords)
# print("Stopwords removed..")

# ===============================================================================>>>Lemmatization
print("Training classifier...")
# print("Lemmatizing..")
nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])
reviews_lemmatized = pd.DataFrame()
reviews_lemmatized['Id'] = English_reviews['Id']
reviews_lemmatized['Label'] = English_reviews['Label']
reviews_lemmatized['Updated Reviews'] = English_reviews['Review'].apply(lemmatize_text)
print(reviews_lemmatized)

# ===============================================================================>>>Training and Testing Distribution

trainsize = training_datasize
trainsize = int((trainsize / 100) * len(reviews_lemmatized))
train_reviews, test_reviews = train_test_split(reviews_lemmatized, train_size=trainsize, random_state=42)
# print("\ntraining data:", len(train_reviews))
# print("testing data:", len(test_reviews))
x_train = train_reviews[['Updated Reviews']]
y_train = train_reviews[['Label']]

# ===============================================================================>>>Tokenization

x_train_tokens = []
for review in x_train['Updated Reviews']:
    tokens = word_tokenize(str(review))
    x_train_tokens.append(tokens)

all_tokens = []
for sublist in x_train_tokens:
    for token in sublist:
        all_tokens.append(token)
# print("total tokens:",len(all_tokens))
all_distinct_tokens=set(all_tokens)
# print("Size of vocabulary:",len(all_distinct_tokens)," tokens\n")  

# # ===================================================================================>bag of words for each label

token_counts = {token: 0 for token in all_distinct_tokens}
for label in range(1, 6):
    label_reviews = train_reviews[train_reviews['Label'] == label]

    for review in label_reviews['Updated Reviews']:
        tokens = word_tokenize(str(review))
        present_tokens = set()
        for token in tokens:
            if token in all_distinct_tokens and token not in present_tokens:
                token_counts[token] += 1
                present_tokens.add(token)

    bag_of_words_label = pd.DataFrame(list(token_counts.items()), columns=['Token', 'Count'])
    # bag_of_words_label.to_csv(f"bag_of_words_label_{label}.csv", index=False)
    # print(f"Bag of words for label {label} saved to bag_of_words_label_{label}.csv")

# # ============================================================================================> Calculating prior probabilities of all labels

prior_probabilities = {}
total_samples = len(y_train)
label_counts = y_train['Label'].value_counts()
total_samples = len(y_train)
prior_probabilities = {}
# print("\n")
for label, count in label_counts.items():
    prior_probability = count / total_samples
    prior_probabilities[label] = prior_probability
    # print(f"Label {label}: Count={count}, Prior Probability={prior_probability:.4f}")
# print("\n")
# print(f"Total number of sentences: {total_samples}")

# ============================================================================================> Calculating token probabilities for each label

token_probabilities = {}
alpha = 1

for label in range(1, 6):
    label_reviews = train_reviews[train_reviews['Label'] == label]
    token_count_label = {token: 0 for token in all_distinct_tokens}
    for review in label_reviews['Updated Reviews']:
        tokens = word_tokenize(str(review))
        for token in tokens:
            if token in all_distinct_tokens:
                token_count_label[token] += 1
    total_tokens_label = sum(token_count_label.values())
    token_probabilities[label] = {}
    for token, count in token_count_label.items():
        probability = (count + alpha) / (total_tokens_label + len(all_distinct_tokens))
        token_probabilities[label][token] = probability

for label, probabilities in token_probabilities.items():
    df = pd.DataFrame(probabilities.items(), columns=['Token', 'Probability'])

    # df.to_csv(f"posterior_probabilities_label_{label}.csv", index=False)
    # print(f"Posterior probabilities for label {label} saved to posterior_probabilities_label_{label}.csv")

# # ============================================================================================> testing
print("Testing Classifier...")
def classify_document(document):
    tokens = word_tokenize(str(document))
    posterior_probabilities = {}
    label_probabilities = {}
   
    for label in range(1, 6):
        posterior_prob = prior_probabilities[label]
        for token in tokens:
            if token in token_probabilities[label]:
                posterior_prob *= token_probabilities[label][token]
        posterior_probabilities[label] = posterior_prob

    predicted_label = max(posterior_probabilities, key=posterior_probabilities.get)
    return predicted_label
# print("\n")
test_reviews['Predicted Label'] = test_reviews['Updated Reviews'].apply(classify_document)
# print(test_reviews[['Updated Reviews', 'Label', 'Predicted Label']])

# ===============================================================================>>>Classifier testing results:

test_reviews['Predicted Label'] = test_reviews['Updated Reviews'].apply(classify_document)
actual_labels = test_reviews['Label']
predicted_labels = test_reviews['Predicted Label']

conf_matrix = confusion_matrix(actual_labels, predicted_labels)
true_positive = (np.diag(conf_matrix))
false_positive = (np.sum(conf_matrix, axis=0) - true_positive)
false_negative = (np.sum(conf_matrix, axis=1) - true_positive)
true_negative = (np.sum(conf_matrix) - (true_positive + false_positive + false_negative))

sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
precision = true_positive / (true_positive + false_positive)
negative_predictive_value = true_negative / (true_negative + false_negative)
accuracy = (true_positive + true_negative) / np.sum(conf_matrix)
precision = np.nan_to_num(precision)
sensitivity = np.nan_to_num(sensitivity)
f_score = 2 * precision * sensitivity / (precision + sensitivity)

print("\nTest Results/Metrics:")

print("True positive: ",true_positive)
print("True negative",true_negative)
print("False positive",false_positive)
print("False negative",false_negative)

print("\n")
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("Negative Predictive Value:", negative_predictive_value)
print("Accuracy:", accuracy)
print("F-score:", f_score)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
# plt.show()

# ============================================================================================> user sentence classifier

def classify_user_document(document):
    tokens = word_tokenize(str(document))
    posterior_probabilities = {}
    label_probabilities = {}
    
    for label in range(1, 6):
        posterior_prob = prior_probabilities[label]
        for token in tokens:
            if token in token_probabilities[label]:
                posterior_prob *= token_probabilities[label][token]
        posterior_probabilities[label] = posterior_prob
    
    total_probability = sum(posterior_probabilities.values())
    for label, posterior_prob in posterior_probabilities.items():
        label_probabilities[label] = posterior_prob / total_probability
    
    predicted_label = max(posterior_probabilities, key=posterior_probabilities.get)
    return predicted_label, label_probabilities

def classify_user_input():
    while True:
        user_sentence = input("Enter a sentence: ")
        predicted_label, label_probabilities = classify_user_document(user_sentence)
        print("\nSentence S:",user_sentence)
        print("was classified as:", predicted_label)
        print("Probabilities for each label:")
        for label, probability in label_probabilities.items():
            print(f"P({label}|{user_sentence}): {probability:.4f}")

        
        response = input("Do you want to enter another sentence [Y/N]? ").strip().upper()
        if response != 'Y':
            break

classify_user_input()
# ============================================================================================> top ten tokens of every label

