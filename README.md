Media Bias:

This program contains machine-learning models that attempt to classify the political bias of a news article based on the actual content of the news article. The program uses a Naive Bayes with a bag-of-words preprocessed by removing stopwords and converting the entire text to lowercase. Then it uses an SVM to classify the text into 3 categories (left, center, and right) corresponding to the respective media bias. Furthermore, to improve the prediction accuracy, we use GridSearchCV() to choose the optimal parameters C, gamma, and kernel, thereby enhancing the performance of the classification. Finally, we implement a CNN by preprocessing the data by converting it to lowercase and removing the ntlk stopwords, followed by tokenizing the input string and passing it through a CNN with 2,581,667 trainable parameters, with early_stopping to avoid overfitting.

Ultimately, the program leverages the idea that news sources themselves have a bias, which can be used to classify the bias of the article, which was done using Naive Bayes, along with bag-of-words and label encoding, leading to a 96.4% accuracy on the testing set. A classifier function is also present at the end of the .ipynb file that classifies an article according to its political bias, using the trained models described above.

Required modules:

    pandas
    numpy
    tensorflow
    nltk
    sklearn

To run the .ipynb file, install the above-mentioned modules after navigating to the file after cloning the repository, and run the .pynb file.

Data obtained from: https://github.com/irgroup/Qbias
