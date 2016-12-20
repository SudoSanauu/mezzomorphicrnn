import unittest
from byte_converter import *

class ByteConverterTestCase(unittest.TestCase):
	
	#testing bytes to int
	#error if no bytes
	def test_expect_bytes(self):
		try:
			bytes_to_int('mum')
			self.fail('expected nonbytes to fail')
		except TypeError:
			pass

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

	#test integer is small enough to fit in our range
	def test_expect_int_in_range(self):
		try:
			int_to_bytes('mum',1)
			self.fail('expected nonint to fail')
		except TypeError:
			pass

		try:
			int_to_bytes(99999999,1)
			self.fail('expected int out of range to fail')
		except ValueError:
			pass

	#test that the int inputed returns correct byte
	def test_one_width_int(self):
		self.assertEqual(int_to_bytes(0,1),b'\x00')
		self.assertEqual(int_to_bytes(255,1),b'\xff')

	#test that int return the correct byte pair
	def test_two_width_int(self):
		self.assertEqual(int_to_bytes(0,2),b'\x00\x00')
		self.assertEqual(int_to_bytes(65535,2),b'\xff\xff')
		self.assertEqual(int_to_bytes(65280,2),b'\x00\xff')

	#test to see if int_to_bytes to bytes_to_int
	def test_int_to_bytes_to_int(self):
		self.assertEqual(bytes_to_int(int_to_bytes(65280,2)),65280)







