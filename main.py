from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import load
import logging
import boto3
import numpy as np
import json

logging.getLogger().setLevel(logging.INFO)

S3 = boto3.resource('s3')
VOCABULARY_BUCKET = "org.gear-scanner.data"
MODELS = dict()
CLASSIFIER_PREFIX = "product-classifier"


def lambda_handler(event, context):
    records = event['Records']
    for record in records:
        logging.info("Processing record: " + str(record))
        message = json.loads(record["body"])
        if "brand" in message:
            brand = message["brand"]
            if brand not in MODELS:
                brand = brand.lower()
                model = read_dump(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "model.joblib"))
                products = list(read_lines(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "products.txt")))
                vocabulary = read_lines(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "vocabulary.txt"))
                vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, vocabulary=vocabulary)
                MODELS[brand] = (products, vectorizer, model)
            original_name = classify(brand, message["name"])
            message["originalName"] = original_name
            print(message)


def read_lines(bucket, key):
    obj = S3.Object(bucket, key)
    return map(lambda line: line.decode('utf-8'), obj.get()['Body'].iter_lines())


def read_dump(bucket, key):
    local_path = "/tmp/" + key.replace("/", "-")
    S3.Bucket(bucket).download_file(key, local_path)
    with open(local_path, 'rb') as file:
        return load(file)


def classify(brand, product_name):
    model = MODELS[brand]
    doc_term_matrix = model[1].transform([product_name]).toarray()
    if not np.sum(doc_term_matrix) > 0:
        return product_name
    else:
        n_neighbour = model[2].kneighbors(doc_term_matrix)
        if n_neighbour[0][0][0] == n_neighbour[0][0][2]:
            return product_name
        else:
            return model[0][n_neighbour[1][0][0]]


# if __name__ == "__main__":
#     record = {
#         "Records": [
#             {
#                 "messageId": "19dd0b57-b21e-4ac1-bd88-01bbb068cb78",
#                 "receiptHandle": "MessageReceiptHandle",
#                 "body": u"""{
#                     "brand": "petzl",
#                     "name": "lynx crampons"
#                 }""",
#                 "attributes": {
#                     "ApproximateReceiveCount": "1",
#                     "SentTimestamp": "1523232000000",
#                     "SenderId": "123456789012",
#                     "ApproximateFirstReceiveTimestamp": "1523232000001"
#                 },
#                 "messageAttributes": {},
#                 "md5OfBody": "7b270e59b47ff90a553787216d55d91d",
#                 "eventSource": "aws:sqs",
#                 "eventSourceARN": "arn:aws:sqs:us-east-2:123456789012:MyQueue",
#                 "awsRegion": "us-east-2"
#             }
#         ]
#     }
#     lambda_handler(record, None)
