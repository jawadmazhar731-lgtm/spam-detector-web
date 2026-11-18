# NOTE (web version only): the ONLY change made to the original file is
# wrapping the interactive `while True: input()` loop at the end in
# `if __name__ == "__main__":`, so this module can be imported by the Flask
# web app (app.py) without blocking on terminal input. All training code,
# clean_text(), predict_message(), the Part A baseline and the Part B
# implementation are EXACTLY as written in the original file.
# Original: ~/workspace/user/files/spam_detector_kaggle_0_4kvq.py

# ============================================================
# SMS SPAM DETECTOR - ML Lab Mids Project
# Zeeshan Ali - BSCS, Iqra University
# ============================================================
# SOURCE OF BASE CODE (as instructed by Sir):
#   Kaggle - "SMS Spam Collection Dataset" (by UCIML on Kaggle,
#   originally from the UCI Machine Learning Repository).
#   Dataset file: spam.csv (5,572 messages, columns v1 = label,
#   v2 = message). The base steps below follow the standard
#   Kaggle notebook for this dataset.
#
# PART A - BASE CODE FROM KAGGLE:
#   Load spam.csv, clean columns, encode labels, split data,
#   CountVectorizer + Multinomial Naive Bayes, check accuracy.
#
# PART B - OUR OWN IMPLEMENTATION (added by us on top of base):
#   1. Text cleaning: clever spellings like "FR33 PR1ZE" are
#      converted back to "FREE PRIZE" before checking.
#   2. Extra local spam examples added to the training data:
#      parcel-delivery scam, SIM-block scam, bank-details scam
#      and Roman Urdu spam (which the Kaggle data does not have).
#   3. TF-IDF with word pairs (bigrams) instead of only
#      CountVectorizer, and ComplementNB (a better Naive Bayes
#      variant for text) instead of plain MultinomialNB.
#   4. Full evaluation: confusion matrix + classification
#      report (base code only prints accuracy).
#   5. Model saving with joblib, and prediction on new
#      messages using the SAVED model (real implementation).
# ============================================================

import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# ---------- Load dataset (Kaggle: spam.csv) ----------
try:
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    print("Loaded Kaggle spam.csv  -> rows:", len(df))
except FileNotFoundError:
    # So the code still runs for demo if spam.csv is not present.
    print("spam.csv not found, using built-in sample dataset.")
    spam_msgs = [
        "WINNER! You have won a free prize. Claim now!",
        "Free entry to win cash, text WIN to 12345",
        "Congratulations! You won a lottery, claim your money now",
        "Get free ringtones now, subscribe today, special offer",
        "Limited offer! Buy now and get 50 percent off, free gift",
        "You have been selected for a free holiday trip, claim now",
        "Earn money fast, click here to claim your cash prize",
        "Free recharge! Claim your free credit now, hurry",
        "Win a brand new iPhone free, hurry up, limited offer",
        "Claim your free gift voucher today only, win cash",
        "Urgent! Your account won a prize, claim money now",
        "Free tickets! You are a winner, text CLAIM to win",
        "Congratulations winner, you won 1000 cash, claim free prize",
        "Special offer, buy one get one free, hurry limited time",
        "You won a free shopping voucher, claim your prize now",
        "Win cash daily, join free and earn money, click now",
        "Lottery winner! Claim your jackpot prize money today",
        "Free bonus! Deposit now and win big cash prize",
        "Hurry! Free offer expires today, claim your gift now",
        "Selected winner, get free credit and cash bonus, claim",
        "Prize alert! You won a free mobile, text WIN now",
        "Earn extra income, free joining, win money fast, click",
        "Congratulations! Free entry into cash prize draw, claim now",
        "Get a free gift card worth 500, limited offer, hurry",
        "You are our lucky winner, claim free holiday prize today",
        "Win free fuel voucher, text CLAIM and get your prize",
        "Urgent offer, lowest price, free delivery, buy now win",
        "Cash prize waiting for you, claim free money now winner",
        "Free trial! Subscribe now and win a special bonus prize",
        "Jackpot! You won cash, claim your free prize immediately",
    ]
    ham_msgs = [
        "Hey, are we meeting tomorrow for the class project?",
        "Please call me when you reach home",
        "I will reach home in 10 minutes, wait for me",
        "Can you send me the class notes from today?",
        "Lunch is ready, come downstairs please",
        "The exam is postponed to next Monday, teacher said",
        "Please submit the assignment by Friday morning",
        "Happy birthday! Hope you have a great day",
        "I am waiting outside your house, come soon",
        "Lets go for a walk in the evening after dinner",
        "Did you complete the homework for tomorrow?",
        "Mom asked you to bring vegetables on your way home",
        "The library book is due next week, please return it",
        "Our group meeting is shifted to 3 pm today",
        "Please bring your notebook to class tomorrow",
        "I left my charger at your place yesterday",
        "Dinner is at 8, do not be late please",
        "The teacher will take attendance in the first lecture",
        "Can we study together at the library this weekend?",
        "Please remind me to pay the electricity bill tomorrow",
        "I am going to the market, do you need anything?",
        "Class is cancelled today because of the rain",
        "Your brother called, call him back when possible",
        "We will play cricket on Sunday morning at the ground",
        "Please forward me the timetable of this semester",
        "I booked the train tickets for our trip next month",
        "Grandma is feeling better now, doctor visited today",
        "The project presentation is scheduled for Thursday",
        "Please water the plants while I am away",
        "See you at the university gate at 9 in the morning",
    ]
    df = pd.DataFrame({
        "label": ["spam"] * len(spam_msgs) + ["ham"] * len(ham_msgs),
        "message": spam_msgs + ham_msgs,
    })

df["label"] = df["label"].map({"ham": 0, "spam": 1})  # ham = 0, spam = 1

# ============================================================
# PART A - BASE CODE (Kaggle style): CountVectorizer + MultinomialNB
# ============================================================
print("\n========== PART A: BASE CODE (Kaggle) ==========")
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42,
    stratify=df["label"],
)
base_vectorizer = CountVectorizer(stop_words="english")
base_model = MultinomialNB()
base_model.fit(base_vectorizer.fit_transform(Xb_train), yb_train)
base_pred = base_model.predict(base_vectorizer.transform(Xb_test))
print("Base Code Accuracy:",
      round(accuracy_score(yb_test, base_pred) * 100, 2), "%")

# ============================================================
# PART B - OUR OWN IMPLEMENTATION
# ============================================================
print("\n========== PART B: OUR IMPLEMENTATION ==========")

# --- B1 + B2: extra local examples added to the training data ---
extra_spam = [
    "Dear customer your parcel could not be delivered, pay the redelivery charge today",
    "Your parcel is held at the warehouse, pay the delivery fee to receive it",
    "Delivery failed, please pay a small charge and confirm your address",
    "Your SIM will be blocked today, confirm your identity on this link",
    "Your bank account will be suspended, verify your password and details now",
    "Share your OTP code to restore your blocked account immediately",
    "I have a business proposal, send me your bank details to transfer the payment",
    "Send your bank account details and receive a large transfer commission",
    "Aap ne inaam jeeta hai, apna number aur CNIC bhejen",
    "Mubarak ho aap lucky winner hain, inaam hasil karne ke liye code bhejen",
    "Ap ka number inaam mein nikla hai, tafseel ke liye rabta karen",
    "Benazir program se paisay milay hain, apna CNIC number bhejen",
]
extra_ham = [
    "The courier delivered your parcel at home today",
    "I received my parcel from the post office yesterday",
    "Aap kal class ayenge ya nahi, bata den",
    "Ammi ne khana bana liya hai, ghar aa jao",
    "Kal university band hai, notification aa gaya hai",
    "Bhai ko mera salaam kehna aur khairiyat poochna",
]
extra_df = pd.DataFrame({
    "label": [1] * len(extra_spam) + [0] * len(extra_ham),
    "message": extra_spam + extra_ham,
})
df2 = pd.concat([df, extra_df], ignore_index=True)
print("Extra local examples added:", len(extra_df), "-> total rows:", len(df2))


# --- B1: text cleaning (undo clever spellings) ---
def clean_text(text):
    def fix_word(word):
        if any(c.isalpha() for c in word) and any(c.isdigit() for c in word):
            table = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                                   "5": "s", "7": "t", "$": "s", "@": "a"})
            return word.translate(table)
        return word

    words = re.findall(r"\S+", str(text).lower())
    return " ".join(fix_word(w) for w in words)


X = df2["message"].apply(clean_text)
y = df2["label"]

# --- B3: TF-IDF (words + pairs) + ComplementNB ---
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)
model = ComplementNB()
model.fit(X_train, y_train)

# --- B4: full evaluation ---
y_pred = model.predict(X_test)
print("\nOur Implementation Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Ham", "Spam"],
                            zero_division=0))

# --- B5: save the model, then predict using the SAVED model ---
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model saved as spam_model.pkl and tfidf_vectorizer.pkl")
saved_model = joblib.load("spam_model.pkl")
saved_vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_message(msg):
    vec = saved_vectorizer.transform([clean_text(msg)])
    result = saved_model.predict(vec)[0]
    return "SPAM" if result == 1 else "HAM (not spam)"


print("\n--- Test messages ---")
for msg in [
    "Congratulations! You won a free prize, claim now",
    "Hey, will you come to class tomorrow?",
    "Dear customer your parcel could not be delivered, pay the redelivery charge today",
    "Aap ne 50000 ka inaam jeeta hai, apna number aur CNIC bhejen",
    "W1n a FR33 pr1ze, c1aim n0w",
]:
    print(f"{msg!r}  ->  {predict_message(msg)}")

if __name__ == "__main__":
    while True:
        user_msg = input("\nEnter a message to check (or press Enter to exit): ")
        if user_msg.strip() == "":
            break
        print("Result:", predict_message(user_msg))
