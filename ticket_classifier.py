import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv("ticket_dataset.csv")

print("Dataset Loaded Successfully!\n")
print(data.head())

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean all tickets
data["clean_text"] = data["ticket_text"].apply(clean_text)

# Features and Labels
X = data["clean_text"]
y = data["category"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel Trained Successfully!")

# Evaluation
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("MODEL PERFORMANCE")
print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# Function to Predict Tickets
def predict_ticket(ticket):

    cleaned = clean_text(ticket)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max() * 100

    if confidence < 60:
        review = "Needs Human Review"
    else:
        review = "Auto Assigned"

    urgent_keywords = [
        "urgent",
        "critical",
        "down",
        "immediately",
        "not working",
        "crash",
        "failed"
    ]

    priority = "Normal"

    ticket_lower = ticket.lower()

    for word in urgent_keywords:
        if word in ticket_lower:
            priority = "High"
            break

    print("Ticket :", ticket)
    print("Predicted Category :", prediction)
    print(f"Confidence : {confidence:.2f}%")
    print("Review Status :", review)
    print("Priority :", priority)

# Predict 5 New Tickets
print("PREDICTING NEW TICKETS")

new_tickets = [

    "Payment failed and refund not received",

    "Laptop not working after software update",

    "Need leave approval for tomorrow",

    "How can I book meeting room?",

    "Urgent server down immediately"

]

for ticket in new_tickets:
    predict_ticket(ticket)

# CLI Demo
print("LIVE TICKET CATEGORIZER")

while True:

    user_ticket = input("\nEnter Ticket (type 'exit' to quit): ")

    if user_ticket.lower() == "exit":
        print("\nProgram Closed.")
        break

    predict_ticket(user_ticket)
