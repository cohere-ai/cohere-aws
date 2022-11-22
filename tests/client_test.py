import unittest

from cohere_sagemaker import Client

class TestClient(unittest.TestCase):

    def setUp(self):
        self.client = Client(endpoint_name='cohere-gpt-medium', region_name='us-east-1')
        super().setUp()

    def tearDown(self):
        self.client.close()

    # TODO unauthorized
    # TODO bad region (endpoint not found)
    # TODO bad variant
    # TODO throw error, don't print

    def test_generate(self):
        # temperature = 0.0 makes output deterministic
        response = self.client.generate("Hello world!", temperature=0.0)
        self.assertEqual(response.generations[0].text, " I'm a newbie to the forum and I'm looking for some help. I have a new")

    def test_generate_variant(self):
        response = self.client.generate("Hello world!", temperature=0.0, variant="p3")
        self.assertEqual(response.generations[0].text, " I'm a newbie to the forum and I'm looking for some help. I have a new")
    

if __name__ == '__main__':
    unittest.main()
