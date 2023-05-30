import numpy
import scipy.special
from PIL import Image

weight_input_hidden = numpy.genfromtxt('weights_input_hidden.csv', delimiter=",")
weight_hidden_output = numpy.genfromtxt('weights_hidden_output.csv', delimiter=",")

input = numpy.asanyarray(Image.open('C:\\Users\\Goldy\\PycharmProjects\\odc\\dataset_preprocessed_test\\0\\hsf_0_01009.png'))
input = input.flatten()
input = (input/255)*0.99 + 0.01

hidden_inputs = numpy.dot(weight_input_hidden, input)
# outputs from hidden layer
hidden_outputs = scipy.special.expit(hidden_inputs)

# inputs to output layer
final_inputs = numpy.dot(weight_hidden_output, hidden_outputs)
# outputs from output layer
final_outputs = scipy.special.expit(final_inputs)

print(numpy.argmax(final_outputs))