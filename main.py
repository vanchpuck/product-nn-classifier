from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import load
import logging
import boto3
import numpy as np
import json
import itertools
from decimal import Decimal

logging.getLogger().setLevel(logging.INFO)


VOCABULARY_BUCKET = "org.gear-scanner.data"
MODELS = dict()
CLASSIFIER_PREFIX = "product-classifier"
PRODUCT_TABLE = "products"


class BrandModelData:

    def __init__(self, brand_products, vectorizer, nn_model):
        self.brand_products = brand_products
        self.vectorizer = vectorizer
        self.nn_model = nn_model


class Classifier:

    def __init__(self, model_bucket):
        self.model_bucket = model_bucket
        self.s3 = boto3.resource('s3')
        self.models = dict()

    def classify(self, brand, product_name):
        if brand not in self.models:
            self.models[brand] = self.read_brand_model_data(brand)
        model: BrandModelData = self.models[brand]
        doc_term_matrix = model.vectorizer.transform([product_name]).toarray()
        if not np.sum(doc_term_matrix) > 0:
            return product_name
        else:
            n_neighbour = model.nn_model.kneighbors(doc_term_matrix)
            if n_neighbour[0][0][0] == n_neighbour[0][0][2]:
                return product_name
            else:
                return model.brand_products[n_neighbour[1][0][0]]

    def read_brand_model_data(self, brand):
        model = self.read_dump(self.model_bucket, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "model.joblib"))
        products = list(self.read_lines(self.model_bucket, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "products.txt")))
        vocabulary = set(itertools.chain.from_iterable([sub.split() for sub in products]))
        vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, vocabulary=vocabulary)
        return BrandModelData(products, vectorizer, model)

    def read_lines(self, bucket, key):
        obj = self.s3.Object(bucket, key)
        return map(lambda line: line.decode('utf-8'), obj.get()['Body'].iter_lines())

    def read_dump(self, bucket, key):
        local_path = "/tmp/" + key.replace("/", "-")
        self.s3.Bucket(bucket).download_file(key, local_path)
        with open(local_path, 'rb') as file:
            return load(file)


VOCABULARY_BUCKET = "org.gear-scanner.data"
CLASSIFIER = Classifier(VOCABULARY_BUCKET)
DYNAMO_DB = boto3.resource('dynamodb')


def lambda_handler(event, context):
    records = event['Records']
    for record in records:
        logging.info("Processing record: " + str(record))
        attempt = json.loads(record["body"])
        classify(attempt)
        write_to_db(PRODUCT_TABLE, attempt)
        # if is_valid(body):
        #     doc = body["document"]
        #     brand = doc["brand"].lower()
        #     name = doc["name"].lower()
        #     original_name = CLASSIFIER.classify(brand, name)
        #     doc["originalName"] = original_name
        #     print(doc)


def classify(attepmt):
    if is_successful_attempt(attepmt):
        doc = attepmt["document"]
        brand = doc["brand"].lower()
        name = doc["name"].lower()
        # TODO Consider to move try to one layer above
        try:
            original_name = CLASSIFIER.classify(brand, name)
            doc["originalName"] = original_name
        except Exception as exc:
            logging.warning("Can't classify product: " + str(exc))


def is_successful_attempt(attempt):
    return "document" in attempt and "name" in attempt["document"] and "brand" in attempt["document"]


# def read_lines(bucket, key):
#     obj = S3.Object(bucket, key)
#     return map(lambda line: line.decode('utf-8'), obj.get()['Body'].iter_lines())
#
#
# def read_dump(bucket, key):
#     local_path = "/tmp/" + key.replace("/", "-")
#     S3.Bucket(bucket).download_file(key, local_path)
#     with open(local_path, 'rb') as file:
#         return load(file)
#
#
#
#
#
def write_to_db(table_name, product):
    logging.info("Saving record: " + str(product))
    table = DYNAMO_DB.Table(table_name)
    if "document" in product:
        document = product["document"]
        if "price" in document:
            document["price"] = Decimal(product["document"]["price"])
    table.put_item(Item=product)
#
#
# if __name__ == "__main__":
#     record = {
#         "Records": [
#             {'messageId': '208271d8-08bc-4c29-84d0-4a4525567734', 'receiptHandle': 'AQEBy9DNryyc9X4X0G4xfBgc4fSfC11bgei2/LRkGZRSAnYLv/Bqbg2ZsAiQPu8VU2ha6Ay9Tl9U8iOCUM3SR4OT8D0w2GhkVBpTGGBqL7eUKJW0r8pmnbWTfF58M94T8rYxdQF9t2DjdyUhid8Uu4oEgs5DrwzI2jq/O0ctG6u2YnXm9fXgbmbGvPAng64tjjvR3oluWtmI/wyjnERgEZ3o7CaoYhB+DJu25MIpMsYBSPyvIm9hyAV+2cbPqNoSfdFFgwqtCslaQxzOikVxNZ+wmfSI7eg2ItbL3y8WY7feeuF2hLrq67Qr/Y4PBdc1YDgCPRXQKFkLRGz6ctjFvuNn03dpbEp6NiVlGZPvrjAvrHZ5B/HGQ+8RjhdkODt4KZmpc12+QMk8PAZf1uoz8+04rQ==', 'body': '{"url":"https://tramontana.ru/product/petzl_koshki_lynx_/","timestamp":1597872559908,"httpCode":200,"responseTime":413,"document":{"url":"https://tramontana.ru/product/petzl_koshki_lynx_/","store":"tramontana.ru","brand":"petzl","name":"Кошки PETZL vasak","category":["Альпинизм и скалолазание","Ледовое снаряжение","Кошки"],"price":19200.0,"currency":"Руб.","imageUrl":"https://tramontana.ru/upload/iblock/f26/1cfa4e46_ab0e_11e2_9a56_005056c00008_521a15cd_92f6_11e6_ab13_18f46a4070d6.jpeg"}}', 'attributes': {'ApproximateReceiveCount': '2', 'SentTimestamp': '1597872560736', 'SenderId': 'AIDA4LFQAY52OP7QXW4Q5', 'ApproximateFirstReceiveTimestamp': '1597872560742'}, 'messageAttributes': {}, 'md5OfBody': 'c88eaeb1222f2db974dde66f25d71962', 'eventSource': 'aws:sqs', 'eventSourceARN': 'arn:aws:sqs:us-east-2:848625190772:ProductClassifierQueue', 'awsRegion': 'us-east-2'}
#         ]
#     }
#     lambda_handler(record, None)
