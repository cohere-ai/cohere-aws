import unittest

from cohere_sagemaker import Client, CohereError

class TestClient(unittest.TestCase):

    def setUp(self):
        self.client = Client(endpoint_name='cohere-gpt-medium', region_name='us-east-1')
        super().setUp()

    def tearDown(self):
        self.client.close()

    # TODO unauthorized

    def test_generate(self):
        # temperature = 0.0 makes output deterministic
        response = self.client.generate("Hello world!", temperature=0.0)
        self.assertEqual(response.generations[0].text, " I'm a newbie to the forum and I'm looking for some help. I have a new")

    def test_variant(self):
        response = self.client.generate("Hello world!", temperature=0.0, variant="AllTraffic")
        self.assertEqual(response.generations[0].text, " I'm a newbie to the forum and I'm looking for some help. I have a new")
    
    def test_max_tokens(self):
        response = self.client.generate("Hello world!", temperature=0.0, max_tokens=40)
        self.assertEqual(response.generations[0].text, " I'm a newbie to the forum and I'm looking for some help. I have a new build PC and I'm trying to get it to work with my TV. I have a Samsung UE")

    def test_bad_region(self):
        client = Client(endpoint_name='cohere-gpt-medium', region_name='invalid-region')
        try:
            client.generate("Hello world!")
            self.fail("expected error")
        except CohereError as e:
            self.assertIn("Could not connect to the endpoint URL", str(e.message))
        client.close()

    def test_wrong_region(self):
        client = Client(endpoint_name='cohere-gpt-medium', region_name='us-east-2')
        try:
            client.generate("Hello world!")
            self.fail("expected error")
        except CohereError as e:
            self.assertIn("Endpoint cohere-gpt-medium of account 455073351313 not found.", str(e.message))
        client.close()

    def test_bad_variant(self):
        try:
            self.client.generate("Hello world!", variant="invalid-variant")
            self.fail("expected error")
        except CohereError as e:
            self.assertIn("Variant invalid-variant not found for Request", str(e.message))

    def test_client_not_connected(self):
        client = Client(region_name='us-east-1')
        try:
            client.generate("Hello world!")
            self.fail("expected error")
        except CohereError as e:
            self.assertIn("No endpoint connected", str(e.message))

if __name__ == '__main__':
    unittest.main()
