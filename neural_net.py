import numpy as np

def sig(sig_input, der=False):
	if der == True:
		return sig_input*(1-sig_input)
	else:
		return 1/(1+np.exp(-sig_input))


def main():
	#creating inputs values
	X = np.array([[0,0,1],
				  [1,1,1],
				  [1,0,1],
				  [0,1,1]])
	y = np.array([[0],[1],[1],[0]])
	learning_r = 0.1
	#assigining the weights randomly
	#first create the seed
	np.random.seed(1)
	#the number of weights is 3 in this case. Random numbers are mean 0, -1 to 1.
	weights = 2* np.random.rand(3,1) - 1

	for i in range(1000):
		#forward propagation
		i_0 = X
		i_1 = sig(np.dot(i_0,weights))

		#calculate the error
		error = y - i_1

		#calculate the delta value
		delta = learning_r * np.dot(i_0.T, (error * sig(i_1, True)))

		#update weights
		weights += delta
		if i % 100 == 0:
			print "complete", (10*i)/100, " %"
			
	print i_1

if __name__ == "__main__":
	main()