def bytes_to_int(input_bytes):
	int_array = []
	for index in range(len(input_bytes)):
		int_array.append(input_bytes[index])
	int_array.reverse()

	final_value = 0
	for index in range(len(int_array)):
		final_value += int_array[index] * (256 ** index)

	return final_value


