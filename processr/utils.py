import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "0", 1: "1"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "variance": d.variance,
            "skewness": d.skewness,
            "kurtosis": d.kurtosis,
            "entropy": d.entropy,
            "bank_note_class": d.bank_note_class,
        }
        for d in data
    ]

    return processed
