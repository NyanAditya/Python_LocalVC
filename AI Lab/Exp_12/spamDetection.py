import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset from URL [cite: 59, 60]
# The manual uses the SMS Spam Collection dataset from a specific GitHub URL [cite: 61]
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

# 2. Encode labels: ham -> 0, spam -> 1 [cite: 62, 63]
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# 3. Split the dataset [cite: 64]
# Splitting into training and testing sets (80% train, 20% test) [cite: 65]
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], 
    data['label_num'], 
    test_size=0.2, 
    random_state=42
)

# 4. Convert text messages into feature vectors 
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 5. Train the Naïve Bayes classifier [cite: 68]
clf = MultinomialNB() # [cite: 70]
clf.fit(X_train_counts, y_train) # [cite: 71]

# 6. Make predictions on test data [cite: 72]
y_pred = clf.predict(X_test_counts) # [cite: 74]

# 7. Evaluate performance [cite: 75]
print("Accuracy:", accuracy_score(y_test, y_pred)) # [cite: 76]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # [cite: 77]