import unittest
from byte_converter import *

class ByteConverterTestCase(unittest.TestCase):

	def test_bytes_to_int(self):
		print("takes in a byte and returns and integer")
		self.assertEqual(type(bytes_to_int(b'\x80')),type(1))