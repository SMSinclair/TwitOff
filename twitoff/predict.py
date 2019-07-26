from .models import User
from twitter import BASILICA
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

def predict_user(user1_name, user2_name, tweet_text):
    """Determine which user is more likely to have authored a tweet"""

    user_set = pickle.dumps((user1_name, user2_name)) #  users sorted
    user1 = User.query.filter(User.name == user1_name).one()
    user2 = User.query.filter(User.name == user2_name).one()
    user1_embeddings = np.array([tweet.embedding for tweet in user1.tweets])
    user2_embeddings = np.array([tweet.embedding for tweet in user2.tweets])
    # X value / features
    embeddings = np.vstack([user1_embeddings, user2_embeddings])
    #  y value / training labels are binary coded
    labels = np.concatenate([np.ones(len(user1.tweets)),
                             np.zeros(len(user2.tweets))])
    #  fit the model (explore best method in a notebook prior)
    log_reg = LogisticRegression().fit(embeddings, labels)
    #  hit the API to get the embedding for the tweet we are predicting
    tweet_embedding = BASILICA.embed_sentence(tweet_text, model="twitter")
    #  logistic regression that returns binary label 1 for user1 0 for user 2
    # returns % likelyhood of both users
    return log_reg.predict(np.array(tweet_embedding).reshape(1, -1))