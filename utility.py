def calculate_mse(x1_list, x2_list):
	size = len(x1_list)
	total_error = 0
	for x1, x2 in zip(x1_list, x2_list):
		err = x2 - x1
		err2 = err ** 2
		total_error += err2
	mse = total_error/size
	return mse
