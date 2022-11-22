import unittest

from cohere_sagemaker import Client

class TestClient(unittest.TestCase):

    def test_generate(self):
        client = Client(endpoint_name='cohere-gpt-medium')
        response = client.generate("Hello world!")
        # TODO actual output assertions
        self.assertEqual(response['ContentType'], 'application/json')
        self.assertEqual(response['InvokedProductionVariant'], 'AllTraffic')

        response = client.generate("Hello world!", variant="p3")
        self.assertEqual(response['ContentType'], 'application/json')
        self.assertEqual(response['InvokedProductionVariant'], 'p3')


if __name__ == '__main__':
    unittest.main()
