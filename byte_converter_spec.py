import unittest
from byte_converter import *

class ByteConverterTestCase(unittest.TestCase):
	#testing bytes to int
	def test_bytes_to_int(self):
		#takes in a byte and returns and integer
		self.assertEqual(type(bytes_to_int(b'\x80')),type(1))

	def test_single_byte(self):
		#successfully converts single byte
		self.assertEqual(bytes_to_int(b'\xff'),255)
		self.assertEqual(bytes_to_int(b'\x00'),0)

	def test_two_bytes(self):
		#successfully converts two bytes
		self.assertEqual(bytes_to_int(b'\x00\x00'),0)
		self.assertEqual(bytes_to_int(b'\xff\xff'),65535)
		self.assertEqual(bytes_to_int(b'\x00\xff'),65280)

	#testing int to bytes
	def test_int_to_byte(self):
		#takes int and returns a byte
		self.assertEqual(type(int_to_bytes(200,1)),type(b''))

