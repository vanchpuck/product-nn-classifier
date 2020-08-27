from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import unittest
import itertools
import mock
import main
from main import PRODUCT_TABLE
import json
import boto3
from botocore.exceptions import ClientError


PRODUCTS = ["lynx", "vasak", "sarken"]
main.DYNAMO_DB = boto3.resource('dynamodb', region_name="us-east-2", endpoint_url="http://localhost:8000")


def mock_read_lines(bucket, key):
    return PRODUCTS


def mock_read_lines_exception(bucket, key):
    raise Exception("Can't read file")


def mock_read_dump(bucket, key):
    vocabulary = [sub.split() for sub in PRODUCTS]
    vocabulary = set(itertools.chain.from_iterable(vocabulary))
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, vocabulary=vocabulary)
    term_document_matrix = vectorizer.fit_transform(PRODUCTS)
    nn_model = NearestNeighbors(n_neighbors=3)
    nn_model.fit(term_document_matrix.toarray())
    return nn_model


class TestRandom(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.create_product_table()

    @classmethod
    def tearDown(cls):
        cls.delete_product_table()

    @classmethod
    def delete_product_table(cls):
        table = main.DYNAMO_DB.Table(PRODUCT_TABLE)
        table.delete()

    @classmethod
    def create_product_table(cls):
        try:
            main.DYNAMO_DB.create_table(
                TableName=PRODUCT_TABLE,
                KeySchema=[
                    {
                        'AttributeName': 'url',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'url',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 10,
                    'WriteCapacityUnits': 10
                }
            )
        except ClientError:
            pass

    @mock.patch('main.Classifier.read_lines', side_effect=mock_read_lines)
    @mock.patch('main.Classifier.read_dump', side_effect=mock_read_dump)
    def test_classifier(self, function1, function2):
        classifier = main.Classifier("dummy_bucket")
        assert "vasak" == classifier.classify("petzl", "VASAK crampon")
        assert "lynx" == classifier.classify("petzl", "lynx modular crampon")
        assert "petzl irvis" == classifier.classify("petzl", "petzl irvis")
        assert "rambo 4" == classifier.classify("grivel", "rambo 4")

    @mock.patch('main.Classifier.read_lines', side_effect=mock_read_lines_exception)
    @mock.patch('main.Classifier.read_dump', side_effect=mock_read_dump)
    def test_handler_exception(self, function1, function2):
        irvis = {
            "url": "https://tramontana.ru/g14/",
            "timestamp": 1597872559908,
            "httpCode": 200,
            "responseTime": 413,
            "document": {
                "url": "https://tramontana.ru/g14/",
                "store": "tramontana.ru",
                "brand": "grivel",
                "name": "Grivel G14",
                "category": ["Альпинизм и скалолазание", "Ледовое снаряжение", "Кошки"],
                "price": 19200.0, "currency": "Руб.",
                "imageUrl": "https://tramontana.ru/g14.jpeg"
            }
        }
        message = {
            "Records": [
                {'messageId': '208271d8-08bc-4c29-84d0-4a4525567734', 'body': json.dumps(irvis)}
            ]
        }
        main.lambda_handler(message, None)
        table = main.DYNAMO_DB.Table(PRODUCT_TABLE)
        actual_irvis = table.get_item(Key={"url": "https://tramontana.ru/g14/"})["Item"]
        assert actual_irvis == irvis

    @mock.patch('main.Classifier.read_lines', side_effect=mock_read_lines)
    @mock.patch('main.Classifier.read_dump', side_effect=mock_read_dump)
    def test_handler(self, function1, function2):
        vasak = {
            "url": "https://tramontana.ru/vasak/",
            "timestamp": 1597872559908,
            "httpCode": 200,
            "responseTime": 413,
            "document": {
                "url": "https://tramontana.ru/product/vasak/",
                "store": "tramontana.ru",
                "brand": "petzl",
                "name": "Кошки PETZL vasak",
                "category": ["Альпинизм и скалолазание", "Ледовое снаряжение", "Кошки"],
                "price": 19200.0, "currency":"Руб.",
                "imageUrl": "https://tramontana.ru/vasak.jpeg"
            }
        }
        lynx = {
            "url": "https://tramontana.ru/lynx/",
            "timestamp": 1597872559909,
            "httpCode": 200,
            "responseTime": 415,
            "document": {
                "url": "https://tramontana.ru/product/lynx/",
                "store": "tramontana.ru",
                "brand": "PETZL",
                "name": "Petzl lynx modular crampons",
                "category": ["Альпинизм и скалолазание", "Ледовое снаряжение", "Кошки"],
                "price": 20200.0, "currency":"Руб.",
                "imageUrl": "https://tramontana.ru/lynx.jpeg"
            }
        }
        message = {
            "Records": [
                {'messageId': '208271d8-08bc-4c29-84d0-4a4525567734', 'body': json.dumps(vasak)},
                {'messageId': '208271d8-08bc-4c29-84d0-4a4525567734', 'body': json.dumps(lynx)}
            ]
        }
        main.lambda_handler(message, None)
        table = main.DYNAMO_DB.Table(PRODUCT_TABLE)

        actual_vasak = table.get_item(Key={"url": "https://tramontana.ru/vasak/"})["Item"]
        self.decode_actual(actual_vasak)
        self.add_original_name(vasak, "vasak")
        assert actual_vasak == vasak

        actual_lynx = table.get_item(Key={"url": "https://tramontana.ru/lynx/"})["Item"]
        self.decode_actual(actual_lynx)
        self.add_original_name(lynx, "lynx")
        assert actual_lynx == lynx

    @classmethod
    def decode_actual(cls, actual_one):
        actual_one["timestamp"] = int(actual_one["timestamp"])
        actual_one["responseTime"] = int(actual_one["responseTime"])
        actual_one["httpCode"] = int(actual_one["httpCode"])
        actual_one["document"]["price"] = float(actual_one["document"]["price"])

    @classmethod
    def add_original_name(cls, attempt, original_name):
        attempt["document"]["originalName"] = original_name
