import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Neurons:

    def __init__(self,input_node_size, hidden_node_size, output_node_size):

        """here using the constructor function to initilize the values required in this network
        most probebly here used self and a common thing because we can access the required values in the specific area without any errors """

        self.lambd_a = 1
        self.learning_rate = 0.01
        self.momentum = 0.8
        self.weights1 = np.random.rand(input_node_size, hidden_node_size)
        self.weights2 = np.random.rand(hidden_node_size, output_node_size)
        # self.train_rmse = float('inf') 
        self.test_rmse = float('inf')
        self.training_loss = []
        self.testing_loss = []
        self.range = []


    def data_processing(self):

      """here in this data processing function it will normilize the data that imported of generated from the game"""

      data = pd.read_csv('data_orginal.csv')
      data = pd.DataFrame(data)
      data = data.dropna()
      data.columns = ['input1', 'input2', 'output1', 'output2'] #this inilize the colume name according to our convinent in my case i used input1 input2 output1 output2

      for column_specific in data.columns:

          # the below condition checks that all the values contains in data sheet is int and float
          if data[column_specific].dtype in ['float64', 'int64']:
              minimum_value = data[column_specific].min()
              maximum_value = data[column_specific].max()
              data[column_specific]=(data[column_specific] - minimum_value)/(maximum_value - minimum_value)

      return data

    def data_segmentation(self):
      """In this data segmentation function it will segment data according to our need here i used 70% for training and 30% fro testing"""
      data = FNN.data_processing()
      self.data = data.sample(frac=1, random_state=42).reset_index(drop=True)

      # Separate Data Based on Percentage
      train_percentage = 0.70
      test_percentage = 0.30
      # val_percentage = 0.15

      num_rows = len(data)

      """so the below two line will give the values like range how much rows need to access for training and testing"""
      num_training_rows = int(train_percentage * num_rows)
      num_testing_rows = int(test_percentage * num_rows)

      training_data = data[:num_training_rows]
      remaining_data = data[num_training_rows:]

      testing_data = remaining_data[:num_testing_rows]
      # val_data = remaining_data[num_test_rows:]

      # below lines is used to inillize the datas using the self as instance of class
      self.training_input = training_data[['input1', 'input2']].values
      self.training_output = training_data[['output1', 'output2']].values

      self.testing_input = testing_data[['input1', 'input2']].values
      self.testing_output = testing_data[['output1', 'output2']].values

    def variance_function(self, inputs, weights):
      """This variance function calculate for the inputes * weights as v to get multiply with lambda"""
      variance = np.dot(inputs, weights)
      return variance

    def front_propogation(self, input_x):

      """in this front propogation function if will calculate the hidden & obtained output and
         returns the value as self.obtained output to the function call """

      variance1 = self.variance_function(input_x, self.weights1) #it will call the variance function by passing the values
      power = (np.dot(-self.lambd_a, variance1)) #this will make dot product of the two matrix
      hidden_input = 1 / (1 + np.exp(power)) #by using this we can find the hidden input values by substuting the previous obtained values in formula
      variance2 = self.variance_function(hidden_input, self.weights2) #it will call the variance function by passing the values
      power = np.dot(-self.lambd_a, variance2) #this will make dot product of the two matrix
      obtained_output = 1 / (1 + np.exp(power)) #by using this we can find the hidden input values by substuting the previous obtained values in formula
      self.hidden_input = hidden_input
      self.obtained_output = obtained_output


      return self.obtained_output


    def back_propogation(self):
      """in this bach propogation function we are find the errors, output gradient, hidden gradient and by using
         output gradient and hidden gradient we are updating the weights by adding the old weights and new weights
         by adding in the same variables weights1 and weights2"""
      self.error = self.training_output - self.obtained_output
      self.output_gradient = self.lambd_a * self.obtained_output * (1 - self.obtained_output) * self.error #we are substuting the obtained values in the formula to find the output gradient

      """we are substuting the obtained values in the formula to find the hidden gradient and
       while finding the hidden gradioent we need to use weights2(hidden to output), it should be in transposed valuev of old weights"""

      self.hidden_gradient = self.lambd_a * self.hidden_input * (1 - self.hidden_input) * self.output_gradient.dot(self.weights2.T)

      """ here we are substuting in the formula to update the weights and we should use the hidden inputs to calculate the
      weights2(hidden to output) and the hidden input should be transposed value, same applies for weights1(input to hidden)
      here instead of hidden input we are using training input"""

      self.weights2 += self.hidden_input.T.dot(self.output_gradient) * self.learning_rate * self.momentum
      self.weights1 += self.training_input.T.dot(self.hidden_gradient) * self.learning_rate * self.momentum


    def loss(self,epoch):

      """in this loss function it act as main function in this case we are finding the training loss(rmse) & testing loss(rmse)
      and ploting the graph for both losses, mainly in this funtion runs as a loop to call the function for front propogation and bach propogation"""

      FNN.data_segmentation()
      """the below for loop runs in range of respective epoch values, in this for loop only the function call occurs"""
      for i in range(epoch):
        obtained_output = self.front_propogation(self.training_input)
        self.back_propogation()

        # self.training_loss.append(np.sqrt(np.mean(np.square( obtined_output - self.output / len(self.input)))))

        #the below line of code calculate training loss(rsme) by substuting in the formula
        y = ((np.sqrt(np.mean(np.square(obtained_output - self.training_output)))))
        self.training_loss.append(y)

        testing_obtained_output=  FNN.front_propogation(self.testing_input)#this line call the front propogation function to calculate output for the testing time
        # self.validation_loss.append(np.sqrt(np.mean(np.square(vallidate_obtained_output - self.validation_output/len(self.validation_input)))))

        #the below line of code calculate testing loss(rsme) by substuting in the formula
        x = (np.sqrt((np.mean(np.square(testing_obtained_output - self.testing_output)))))
        self.testing_loss.append(x)
        if i % 100 == 0:
            print(f"Epoch={i} : Training Loss={y} : Testing loss={x}")

        """For certain reasons i've commanded the earlyg stopping cretria if this 
        condition need to run just uncommand the below three lined"""
        # if x > self.test_rmse:
        #    break
        # self.test_rmse = x
      FNN.graph()

    def graph(self):
      """in this function we are ploting the graph using matplotlib
      the respective graph ploted for training loss and testing loss"""
      print(f"Weights Inputs to Hidden:\n {self.weights1}")
      print(f"Weights Hidden to Output:\n {self.weights2}")

      plt.plot(self.training_loss, label='Training Loss')
      plt.plot(self.testing_loss, label='Testing Loss')
      plt.title('Training & Testing Loss over Epochs')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.show()

epoch = 1000
input_node_size = 2
hidden_node_size = 5
output_node_size =2
FNN = Neurons(input_node_size, hidden_node_size, output_node_size)
FNN.loss(epoch)