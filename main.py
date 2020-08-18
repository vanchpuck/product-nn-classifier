from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import load
import logging
import boto3
import numpy as np

logging.getLogger().setLevel(logging.INFO)

S3 = boto3.resource('s3')
VOCABULARY_BUCKET = "org.gear-scanner.data"
MODELS = dict()
CLASSIFIER_PREFIX = "product-classifier"


def lambda_handler(event, context):
    records = event['Records']
    for record in records:
        logging.info("Processing record: " + str(record))
        message = record["body"]
        if "brand" in message:
            brand = message["brand"]
            if brand not in MODELS:
                brand = brand.lower()
                model = read_dump(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "model.joblib"))
                products = read_lines(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "products.txt"))
                vectorizer = read_dump(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "vectorizer.joblib"))
                MODELS[brand] = (products, vectorizer, model)
            original_name = classify(brand, message["name"])
            message["originalName"] = original_name
            print(message)


def read_lines(bucket, key):
    obj = S3.Object(bucket, key)
    return list(map(lambda line: line.decode('utf-8'), obj.get()['Body'].iter_lines()))


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
#     brand = "petzl".lower()
#     nn_model = read_dump(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "model.joblib"))
#     products = read_lines(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "products.txt"))
#     vectorizer = read_dump(VOCABULARY_BUCKET, "{}/{}/{}".format(CLASSIFIER_PREFIX, brand, "vectorizer.joblib"))
#     MODELS[brand] = (products, vectorizer, nn_model)
#     original_name = classify(brand, "petzl lynx")
#     print(original_name)
#     # vm_pair = read_dump(VOCABULARY_BUCKET, "product-classifier/model/petzl")
#     # # execute only if run as a script
#     # # vm_pair = create_model("petzl")
#     # doc_term_matrix = vm_pair[1].transform(["lynx crampon"]).toarray()
#     # nearest = vm_pair[2].kneighbors(doc_term_matrix)
#     #
#     # if nearest[0][0][0] == nearest[0][0][2]:
#     #     print("unknown2")
#     # else:
#     #     print(vm_pair[0][nearest[1][0][0]])
