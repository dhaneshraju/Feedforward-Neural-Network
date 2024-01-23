import numpy as np
import ast
class NeuralNetHolder:

    def __init__(self):
        # super().__init__()
        self.weights_h = [[ 0.17185705, -0.38970377,  0.15747586, -0.32987734, -0.97894831, -0.21141296,
                           -0.19247314, -0.30119317,  0.68443659, -0.98471554],[-0.21888276, -0.13611006,  0.0495602,  -0.36075413, -0.1608491,   0.4015104,
                                                                             -0.11072151, -0.00778133,  0.45835375, -0.82133583]]
        
        self.weights_o =  [[-5.62919014, -4.45945534],
                           [-5.65565989, -4.33361765],
                           [-5.87206295, -4.47427598],
                           [-4.99918574, -4.3202599 ],
                           [-5.07774774, -4.15563954],
                           [-5.97708094, -3.9519914 ],
                           [-5.61695856, -3.9946133 ],
                           [-5.87290129, -3.92900212],
                           [-6.11003262, -5.00016036],
                           [-4.68927015, -3.76131092]]
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






