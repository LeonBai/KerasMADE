import numpy as np

#building Tree-Like MRF with inputsize number of variables
inputsize = 4
num_of_params = 3
num_of_train_samples = 300
num_of_valid_samples = 100
num_of_test_samples = 100
#for a very elemntry testing i considered graph shape like this:
#			*
#			|
#			*
#		   / \
#		  *   *
#
params = np.random.sample(3)

# Here i'm trying to make samples as data set relevant to random parameters
# for this tree
all_outcomes = np.ndarray(shape=(2**inputsize, inputsize), dtype=np.float32)
prob_of_outcomes = []
for i in range(2**inputsize):
	str_samp = '{0:04b}'.format(i)
	asarr_samp = [int(d) for d in str_samp]
	all_outcomes[i] = asarr_samp
	p = np.exp(
	params[0]*(all_outcomes[i][0])*(all_outcomes[i][1]) \
	+ params[1]*(all_outcomes[i][1])*(all_outcomes[i][2]) \
	+ params[2]*(all_outcomes[i][1])*(all_outcomes[i][3])
	)
	prob_of_outcomes.append(p)

prob_of_outcomes[:] = [x/sum(prob_of_outcomes) for x in prob_of_outcomes]

cum_probs = []
s = 0
for x in prob_of_outcomes:
	s = s + x
	cum_probs.append(s)

train_data = np.ndarray(shape=(num_of_train_samples, inputsize), dtype=np.float32)
train_data_probs = np.ndarray(shape=(num_of_train_samples), dtype=np.float32)
for x in range(num_of_train_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    train_data[x] = all_outcomes[i]
    train_data_probs[x] = prob_of_outcomes[i]


valid_data = np.ndarray(shape=(num_of_valid_samples, inputsize), dtype=np.float32)
valid_data_probs = np.ndarray(shape=(num_of_valid_samples), dtype=np.float32)
for x in range(num_of_valid_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    valid_data[x] = all_outcomes[i]
    valid_data_probs[x] = prob_of_outcomes[i]

test_data = np.ndarray(shape=(num_of_test_samples, inputsize), dtype=np.float32)
test_data_probs = np.ndarray(shape=(num_of_test_samples), dtype=np.float32)
for x in range(num_of_test_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    test_data[x] = all_outcomes[i]
    test_data_probs[x] = prob_of_outcomes[i]

np.savez('simple_tree.npz', 
         inputsize=inputsize, 
         train_length=num_of_train_samples,
         train_data=train_data, 
         train_data_probs = train_data_probs, 
         valid_length=num_of_valid_samples, 
         valid_data=valid_data,
         valid_data_probs=valid_data_probs,
         test_length=num_of_test_samples, 
         test_data=test_data,
         test_data_probs=test_data_probs,
         params=params)
