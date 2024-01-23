import numpy as np
import ast
class NeuralNetHolder:

    def __init__(self):
        # super().__init__()
        self.weights_h = [[0.34670316, -4.97192151],
                          [-6.22363625,  0.6822172]]
        self.weights_o = [[3.13147208,  0.95806162],
                          [-1.39352171, -1.91659837]]
        self.lambd_a = 1
        self.input1_min = -778.6349004784918
        self.input1_max = 744.8081759627983
        self.input2_min = 65.578327522481
        self.input2_max = 1018.5733111416724
        self.output1_min = -7.486164686349453
        self.output1_max = 7.999999999999988
        self.output2_min = -7.97109156296624
        self.output2_max = 7.697784685356044


    def variance_function(self, inputs, weights):
      variance = np.dot(inputs, weights)
      return variance
    def forward_propogation(self, input_a):
        variance1 = self.variance_function(input_a,self.weights_h)
        power = (np.dot(-self.lambd_a, variance1))
        hidden_input = 1 / (1 + np.exp(power))
        variance2 = self.variance_function(hidden_input,self.weights_o)
        power = np.dot(-self.lambd_a, variance2)
        obtained_output = 1 / (1 + np.exp(power))
        self.hidden_input = hidden_input
        self.obtained_output = obtained_output

        return self.obtained_output

    def predict(self, input_row):
        input_row = ast.literal_eval(input_row)

        input1 = (input_row[0] - self.input1_min)/(self.input1_max - self.input1_min)
        input2 = (input_row[1] - self.input2_min)/(self.input2_max - self.input2_min)

        input_a = [input1, input2]

        output = self.forward_propogation(input_a)

        denormalized_output1 = output[0] * (self.output1_max - self.output1_min) + self.output1_min
        denormalized_output2 = output[1] * (self.output2_max - self.output2_min) + self.output2_min

        denormalized_output = [denormalized_output1, denormalized_output2]

        return denormalized_output






