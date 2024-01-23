import numpy as np
import ast
class NeuralNetHolder:

    def __init__(self):
        # super().__init__()
        self.weights_h =  [[1.91144946, 0.57183653, 0.94147817, 1.73644229, 2.08967131, 1.53669044,
                            1.88449005, 1.97118228, 1.61202375, 0.76810085],[1.58463388, 0.46205523, 1.16805298, 1.38784405, 1.88823439, 1.52333356,
                                                                         1.50887683, 1.80636469, 1.22754992, 0.5121324]]
        
        self.weights_o =   [[5.04330475, 7.92180868],
                            [4.30043925, 6.29956076],
                            [4.99450408, 7.67543561],
                            [4.44181572, 7.21346318],
                            [5.03454501, 7.72149283],
                            [4.67288729, 7.66890613],
                            [4.83574532, 7.97885204],
                            [4.75942492, 7.5718272 ],
                            [4.80463718, 7.80739754],
                            [4.43575941, 6.31544535]]
        self.lambd_a = 0.9
        self.input1_min = -1490.2684867282424
        self.input1_max = 1549.2286447798426
        self.input2_min = 65.50339173052987
        self.input2_max = 1018.5733111416724
        self.output1_min = -7.952841571430801
        self.output1_max = 7.999999999999988
        self.output2_min = -7.996603920298254
        self.output2_max = 7.998420608105349


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






