import numpy as np

def sig(sig_input, der=False):
	if der == True:
		return sig_input*(1-sig_input)
	else:
		return 1/(1+np.exp(-sig_input))

def single(inputs, output):
	learning_r = 0.1
	#assigining the weights randomly
	#first create the seed
	np.random.seed(1)
	#the number of weights is 3 in this case. Random numbers are mean 0, -1 to 1.
	weights = 2* np.random.rand(3,1) - 1

	for i in range(1000):
		#forward propagation
		i_0 = inputs
		i_1 = sig(np.dot(i_0,weights))

		#calculate the error
		error = output - i_1

		#calculate the delta value
		delta = learning_r * np.dot(i_0.T, (error * sig(i_1, True)))

		#update weights
		weights += delta
		if i % 100 == 0:
			print "complete", (10*i)/100, " %"
			
	return i_1

def oneHidden(inputs, output):
	learning_r = 1

	#initializing the weights; weight_1 and weight_2
	np.random.seed(1)
	weight_1 = 2*np.random.rand(3,4) - 1
	weight_2 = 2*np.random.rand(4,1) - 1

	for i in range(60000):
		#forward propagation
		i_0 = inputs
		i_1 = sig(np.dot(i_0,weight_1))
		i_2 = sig(np.dot(i_1,weight_2))

		#calculate the error at layer i_2
		i_2_error = output - i_2
		if i % 1000 == 0:
			print "complete", (10*i)/100, " %, Error:" + str(np.mean(np.abs(i_2_error)))
		#calculate the delta at i_2 
		i_2_delta = i_2_error*sig(i_2, True)

		#calculat how much each l_1 contribute to the i_2 values
		i_1_error = i_2_delta.dot(weight_2.T)

		#calculate the delta at i_1
		i_1_delta = i_1_error*sig(i_1, True)

		weight_2 += learning_r*np.dot(i_1.T, i_2_delta)
		weight_1 += learning_r*np.dot(i_0.T, i_1_delta)
	
	return i_2
def main():
	#creating inputs values
	X = np.array([[0,0,1],
				  [0,1,1],
				  [1,0,1],
				  [1,1,1]])
	y = np.array([[0],[1],[1],[0]])
	#res = single(X,y)
	res = oneHidden(X,y)
	print res

if __name__ == "__main__":
	main()