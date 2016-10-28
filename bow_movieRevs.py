import pandas as pd
import nltk
# Uncomment the bottom line if you do not already have the nltk corpuses on your system.
# nltk.download()
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

trainingData = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# This function converts raw reviews from the tsv data and processes it according to the definitions in movieRev
def movieRevToWords(unprocessed_review):

    htmlRemoval = BeautifulSoup(unprocessed_review, "html.parser").get_text()

    # Applying regex.
    removePunctuations = re.sub("[^a-zA-Z]", " ", unprocessed_review)

    # Converting to lowercase & tokenizing each word.
    convLowerSplit = removePunctuations.lower().split()

    # Searching for stop words in each review(Faster to iterate through a set rather than through a list)
    stopWords = set(stopwords.words("english"))

    # Removing stopwords in each review
    stopWordFreeReview = [w for w in convLowerSplit if not w in stopWords]

    # Combining the set of words to a string so that the string is no longer tokenized and easily readable.
    return (" ".join(stopWordFreeReview))


# processedReview = movieRevToWords(trainingData["review"][20])
# print processedReview
# output: soylent green one best disturbing science fiction movies still persuasive even today standards although flawed little dated apocalyptic touch environmental premise typical time still feel unsettling thought provoking film quality level surpasses majority contemporary sf flicks strong cast intense sequences personally consider classic new york depressing place alive population unemployment unhealthy climate total scarcity every vital food product form food available synthetic distributed soylent company charlton heston great shape plays cop investigating murder one soylent eminent executives stumbles upon scandals dark secrets script little sentimental times climax really come big surprise still atmosphere tense uncanny riot sequence truly grueling easily one macabre moments cinema edward g robinson ultimately impressive last role great modest supportive role joseph cotton baron blood abominable dr phibes science fiction book nightmarish inevitable fade humanity fancy space ships hairy monsters attacking planet
# print len(processedReview), "words"
# output: 1025 words

numberofReviews2BProcessed = trainingData["review"].size
# print numberofReviews2BProcessed

# This array will hold all the processedReviews
trainingProcessedReviews = []

# Iterate through all the reviews to process them.
for i in xrange(0, numberofReviews2BProcessed):
    if (i+1 % 1000 == 0):
        print "Review %d of %d/n" % (i+1, numberofReviews2BProcessed)
    trainingProcessedReviews.append(movieRevToWords(trainingData["review"][i]))
# print trainingProcessedReviews

# Importing the feature extraction model from scikit.
print "Creating Bag of Words...\n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor= None, stop_words=None, max_features= 5000)

training_data_features = vectorizer.fit_transform(trainingProcessedReviews)

training_data_features = training_data_features.toarray()

print training_data_features.shape
# output : (25000, 5000)

# We now check out the features the Bag of Words model generated.
# vocabulary = vectorizer.get_feature_names()
# print "Vocabulary:", vocabulary
# print "Vocabulary Length:", len(vocabulary)

# Random Forest Algorithm.
from sklearn.ensemble import RandomForestClassifier

# A Random Forest Tree is classified with 100 trees.
rForest = RandomForestClassifier(n_estimators = 100)

rForest = rForest.fit(training_data_features, trainingData["sentiment"])

# Testing the accuracy of the model with test-data.

testData = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
# print testData.shape

numberofReviews = len(testData["review"])

processedTestReviews = []

for x in xrange(0, numberofReviews):
    clean_reviews = movieRevToWords(testData["review"][i])
    processedTestReviews.append(clean_reviews)

test_data_features = vectorizer.transform(processedTestReviews)
test_data_features = test_data_features.toarray()

# Making Sentiment Predictions
result = rForest.predict(test_data_features)

# Now we copy results to pandas dataframe with an id column and a sentiment column
output = pd.DataFrame( data={"id":testData["id"], "sentiment": result})

# Writing the results to a csv file.
output.to_csv("Bag_of_words.csv", index="False", quoting=3)
