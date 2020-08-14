from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import logging
import boto3
import itertools
import numpy as np

S3 = boto3.resource('s3')
VOCABULARY_BUCKET = "org.gear-scanner.data"
MODELS = dict()


# def synonyms_preprocessor(string, synonyms):
#     for i in synonyms:vm_modelpair
#         string = string.replace(i[0], i[1])
#     return string


def lambda_handler(event, context):
    records = event['Records']
    for record in records:
        logging.info("Processing record: " + str(record))
        message = record["body"]
        if "brand" in message:
            brand = message["brand"]
            if brand not in MODELS:
                MODELS[brand] = create_model(brand)
            original_name = classify(brand, message["name"])
            message["originalName"] = original_name
            print(message)


def read_brand_products(brand):
    obj = S3.Object(VOCABULARY_BUCKET, brand.lower() + "-vocabulary.txt")
    return list(map(lambda line: line.decode('utf-8'), obj.get()['Body'].iter_lines()))


def create_model(brand):
    brand_products = read_brand_products(brand)
    vocabulary = [sub.split() for sub in brand_products]
    vocabulary = set(itertools.chain.from_iterable(vocabulary))
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, vocabulary=vocabulary)
    term_document_matrix = vectorizer.fit_transform(brand_products)
    nn_model = NearestNeighbors(n_neighbors=3)
    nn_model.fit(term_document_matrix.toarray())
    return brand_products, vectorizer, nn_model


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
    # # execute only if run as a script
    # vm_pair = create_model("petzl")
    # doc_term_matrix = vm_pair[1].transform(["lynx crampon"]).toarray()
    # nearest = vm_pair[2].kneighbors(doc_term_matrix)
    #
    # if nearest[0][0][0] == nearest[0][0][2]:
    #     print("unknown2")
    # else:
    #     print(vm_pair[0][nearest[1][0][0]])
