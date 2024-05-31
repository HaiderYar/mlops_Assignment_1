import unittest
from app import app


class TestApp(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_basic_input(self):
        result = self.app.post('/index', data=dict(a='3', b='5'))
        self.assertIn(b'Predicted Y for X = [[3.]] is 4.2', result.data)

    def test_large_input(self):
        result = self.app.post('/index', data=dict(a='15', b='25'))
        self.assertIn(b'Predicted Y for X = [[15.]] is 16.6', result.data)

    def test_negative_input(self):
        result = self.app.post('/index', data=dict(a='-4', b='-2'))
        self.assertIn(b'Predicted Y for X = [[-4.]] is -1.8', result.data)

    def test_decimal_input(self):
        result = self.app.post('/index', data=dict(a='2.5', b='3.5'))
        self.assertIn(b'Predicted Y for X = [[2.5]] is 3.3', result.data)

    def test_string_input(self):
        result = self.app.post('/index', data=dict(a='abc',
                                                   b='def'))
        self.assertIn(b'Error', result.data)

    def test_out_of_range_input(self):
        result = self.app.post('/index', data=dict(a='20', b='22'))
        self.assertIn(b'Input value is beyond the trained range', result.data)

    def test_missing_input(self):
        result = self.app.post('/index', data=dict())
        self.assertIn(b'Error', result.data)


if __name__ == '_main_':
    unittest.main()
