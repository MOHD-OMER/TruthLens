import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from PIL import Image
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import pickle


from sklearn.feature_extraction.text import CountVectorizer

def compAlg():
    # load the data
    df_true = pd.read_csv('../True.csv')
    df_fake = pd.read_csv('../Fake.csv')

    # add a target class columlns to indicate wheather the news is real or fake
    df_fake['target'] = 0

    df_true['target'] = 1

    # concatenate the two dataframes
    df = pd.concat([df_true, df_fake]).reset_index(drop=True)

    df = df.drop(columns='date')

    # combine title and text together
    df['content'] = df['title'] + ' ' + df['text']

    # create number of sentences
    df['num_characters'] = df['content'].apply(len)

    # crete number of words
    df['num_words'] = df['content'].apply(lambda x: len(nltk.word_tokenize(x)))

    # create number of sentences
    df['num_sentences'] = df['content'].apply(lambda x: len(nltk.sent_tokenize(x)))

    # Initialize stop words
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Step 1: Convert to lowercase
        text = text.lower()
        # Step 2: Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # Step 3: Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Step 4: Remove special characters, punctuation, and numbers
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        # Step 5: Tokenize text
        words = word_tokenize(text)
        # Step 6: Remove stop words
        words = [word for word in words if word not in stop_words]
        # Step 7: Join words back into a single string
        processed_text = ' '.join(words)

        return processed_text

    df['clean_text'] = df['content'].apply(preprocess_text)

    cv = CountVectorizer()

    x = cv.fit_transform(df['clean_text'])
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier()
    mnb = MultinomialNB()
    dtc = DecisionTreeClassifier(max_depth=5)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    abc = AdaBoostClassifier(n_estimators=50, random_state=2)
    bc = BaggingClassifier(n_estimators=50, random_state=2)
    etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
    gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
    xgb = XGBClassifier(n_estimators=50, random_state=2)

    clfs = {
        'SVC': svc,
        'KN': knc,
        'NB': mnb,
        'DT': dtc,
        'LR': lrc
    }

    def train_classifier(name, clf, x_train, y_train, x_test, y_test):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        percision = precision_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Save as JPG
        plt.savefig(f"../static/vis/cm_{name}.jpg")
        plt.close()

        return accuracy, percision


    accuracy_scores = []
    precision_scores = []

    for name, clf in clfs.items():
        current_accuracy, current_precision = train_classifier(name, clf, x_train, y_train, x_test, y_test)

        print("For ", name)
        print("Accuracy - ", current_accuracy)
        print("Precision - ", current_precision)
        accuracy_scores.append(current_accuracy)
        precision_scores.append(current_precision)

    performance_df = pd.DataFrame(
        {'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Accuracy',
                                                                                                            ascending=False)

    performance_df1 = pd.melt(performance_df, id_vars="Algorithm")

    sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df1, kind='bar', height=5)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation='vertical')
    plt.savefig('../static/vis/CompAlg.jpg')

    # select for voting classifer
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    abc = AdaBoostClassifier(n_estimators=50, random_state=2)
    xgb = XGBClassifier(n_estimators=50, random_state=2)
    bc = BaggingClassifier(n_estimators=50, random_state=2)
    voting = VotingClassifier(estimators=[('LR', lrc), ('AdaBoost', abc), ('xgb', xgb), ('BgC', bc)])

    voting.fit(x_train, y_train)

    model = voting
    voting = model

    y_pred = voting.predict(x_test)
    print('Accuracy:...', accuracy_score(y_test, y_pred))
    print('Precision:...', precision_score(y_test, y_pred))

    pickle.dump(cv, open('../models/vectorizer.pkl', 'wb'))
    pickle.dump(model, open('../models/model.pkl', 'wb'))

#compAlg()










