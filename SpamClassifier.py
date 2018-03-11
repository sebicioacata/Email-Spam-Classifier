import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.svm import LinearSVC as svc
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.externals import joblib


def makeDictionary(trainingDirectory):
    """
        Group all words from the emails into a dictionary
        
        @type trainingDirectory: string
        @param trainingDirectory: the name of the directory containing the files of the data set
        @rtype: dictionary
        @return: the dictionary of all words from the emails
    """
    
    emailList = [os.path.join(trainingDirectory, file) for file in os.listdir(trainingDirectory)]
    wordList = []
    
    currFile = 0 # A file is an email in the emailList.
    
    for email in emailList:
        with open(email) as emailContents: # Replace with open(email, encoded = "Latin-1") for trainingSet2
            for lineNo, line in enumerate(emailContents):
                if lineNo == 2: # The contents of the email are found after the third line for the first data set, and second line for the second one
                    words = line.split()
                    wordList += words
        print("Creating the word dictionary | Iteration", currFile)
        currFile += 1
        
        
    wordDictionary = Counter(wordList) # Uses collections.Counter for turning the list into a dictionary
    
    removeList = list(wordDictionary.keys()) # Casting to list prevents modifying the original dictionary
    
    for item in removeList: # Remove words with non-alphabetical characters and/or a length of 1
        if item.isalpha() == False: 
            del wordDictionary[item]
        elif len(item) == 1:
            del wordDictionary[item]
            
    wordDictionary = wordDictionary.most_common(len(wordDictionary))
    
    return wordDictionary


def extractFeatures(trainingDirectory, wordDictionary):
    """
        Extract a feature matrix of size numberOfTrainingFiles x numberOfWords where
            numberOfTrainingFiles depends on the training set (e.g. for the first training set this is 702)
            numberOfWords is the length of the wordDictionary returned by the makeDictionary function
            
        Each row of the feature matrix is encoded as follows: for every of the len(wordDictionary) words
        the value is either 0 (if the word does not appear in the file) or the number of occurences of the word
        
        @type trainingDirectory: string
        @param trainingDirectory: the name of the directory containing the files of the data set
        @type wordDictionary: dictionary
        @param wordDictionary: the dictionary of all words from the emails
        @rtype: matrix
        @return: the features matrix
    """
    
    emailList = [os.path.join(trainingDirectory, file) for file in os.listdir(trainingDirectory)]
    features = np.zeros((len(emailList), len(wordDictionary)))
    
    currFile = 0 # A file is an email in the emailList. This keeps track of the rows of the features matrix
    
    for file in emailList: 
        with open(file) as emailContents: # Replace with open(file, encoded = "Latin-1") for trainingSet2
            for lineNo, line in enumerate(emailContents): 
                if lineNo == 2: # The contents of the email are found after the third line for the first data set, and second line for the second one
                    words = line.split()
                    for word in words:
                        currWord = 0 # This keeps track of the columns of the features matrix
                        for i, line in enumerate(wordDictionary):
                            if word == line[0]: # If the word is found in the file add its number of occurences to the current row
                                currWord = i
                                features[currFile, currWord] = words.count(word)
            currFile += 1
            print("Creating the features matrix | Iteration", currFile)
        
    return features


def categorizeEmails(trainingDirectory):
    """
        Count the number of emails (total, spam, ham) for labeling
        
        @type trainingDirectory: string
        @param trainingDirectory: the name of the directory containing the files to use for training
    """
    
    emailList = [os.path.join(trainingDirectory, file) for file in os.listdir(trainingDirectory)]
    
    spamEmails = 0
    hamEmails = 0
    
    for email in emailList:
        if "sp" not in str(email):
            hamEmails += 1
        
        else:
            spamEmails += 1
    
    return spamEmails, hamEmails


def train(trainingMatrix, trainingLabels, modelType):
    """
        Returns an sklearn Naive Bayes Classifier model
        
        @type trainingMatrix: matrix
        @param trainingMatrix: the feature matrix for the given training data set
        @type trainingLabels: list
        @param trainingLabels: the list of training files labels
        @rtype: MultinomialNB, LinearSVC
        @return: the @rtype model
    """
    
    model = modelType()
    model.fit(trainingMatrix, trainingLabels)
    
    joblib.dump(model, 'model.pkl') # Saving the trained model for a future load
    
    return model


def loadModel(filename):
    """
        Load a pre-trained model
        
        @type filename: string
        @param filename: the name of the .pkl file
    """
    
    model = joblib.load(filename)
    
    return model;


def test(model, testDirectory, testMatrix, testLabels):
    """
        test module for an sklearn model
        
        @type model: sklearn model
        @param model: a trained model (NB, SVC etc.)
        @type testDirectory: string
        @param testDirectory: the name of the directory containing the files to use for testing
        @type testMatrix: matrix
        @param testMatrix: the feature matrix for the given test data set
        @type testLabels: list
        @param testLabels: the list of test files labels
        @rtype: MultinomialNB or LinearSVC
        @return: the predicting model
    """
    
    resultModel = model.predict(testMatrix)
    print(confusion_matrix(testLabels,resultModel)) # Prints confusion matrix to show prediction accuracy
    
    return resultModel
    

def main():
    # Initializing and training
    trainingDirectory = 'trainingSet1/trainingEmails'
    wordDictionary = makeDictionary(trainingDirectory)
    
    spamEmails, hamEmails = categorizeEmails(trainingDirectory)
    total = spamEmails + hamEmails
    
    # Labeling the training data
    trainingLabels = np.zeros(total)
    trainingLabels[hamEmails:total] = 1
    
    trainingMatrix = extractFeatures(trainingDirectory, wordDictionary)
    
    model = train(trainingMatrix, trainingLabels, nb()) # Or ..., svc()) 
#    modelSVC = trainSVC(trainingMatrix, trainingLabels) # Optional
#    model = loadModel('model.pkl') # Loading an already trained model
    
    # Testing 
    testDirectory = 'trainingSet1/testEmails'
    testMatrix = extractFeatures(testDirectory, wordDictionary)
    
    spamEmails, hamEmails = categorizeEmails(testDirectory)
    total = spamEmails + hamEmails
    
    # Labeling the test data
    testLabels = np.zeros(total)
    testLabels[hamEmails:total] = 1
    
    test(model, testDirectory, testMatrix, testLabels)
    
    
    
if __name__ == "__main__":
    main()
    
