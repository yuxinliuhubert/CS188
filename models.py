import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # print("NN DOT PROCUCT", nn.DotProduct(x, self.w))
        return nn.DotProduct(x, self.w)



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            updated = False
            for x, label in dataset.iterate_once(1):
                actual_label = nn.as_scalar(label)
                if self.get_prediction(x) != actual_label:
                    
                    self.w.update(x,actual_label)
                    updated = True

            if not updated:
                break
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size =512
        self.w1 = nn.Parameter(1, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        # self.w2 = nn.Parameter(hidden_layer_size,hidden_layer_size)

        self.w2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2 = nn.Parameter(1, 1)


        self.w_o = nn.Parameter(hidden_layer_size, 1)
        self.b_o = nn.Parameter(1, 1)
        self.batch_size = 200
        self.learning_rate = -0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        hidden_output1= nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        hidden_output2 = nn.ReLU(nn .Linear(nn.AddBias(hidden_output1,self.b1), self.w2))
        predicted_value = nn.AddBias(nn.Linear(hidden_output2, self.w_o), self.b_o)
        return predicted_value

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_value = self.run(x)
        return nn.SquareLoss(predicted_value, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        while True:
            for x, y in dataset.iterate_forever(self.batch_size):
                loss = self.get_loss(x, y)
                if nn.as_scalar(loss) < 0.02:
                    return
                grad = nn.gradients(loss, [self.w1, self.b1, self.w_o, self.b_o])
                self.w1.update(grad[0], self.learning_rate)
                self.b1.update(grad[1], self.learning_rate)
                self.w_o.update(grad[2], self.learning_rate)
                self.b_o.update(grad[3], self.learning_rate)




class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size = 512
        self.batch_size = 100
        self.learning_rate = -0.5

        self.w1 = nn.Parameter(784, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        # self.w2 = nn.Parameter(hidden_layer_size,hidden_layer_size)

        self.w2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2 = nn.Parameter(1, hidden_layer_size)

        # # self.w2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        # # self.b2 = nn.Parameter(1, 1)

        # self.w3 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        # self.b3 = nn.Parameter(1, hidden_layer_size)

        self.w_o = nn.Parameter(hidden_layer_size, 10)
        self.b_o = nn.Parameter(1, 10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
  
        hidden_output1= nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        
        hidden_output2 = nn.ReLU(nn.AddBias(nn.Linear(hidden_output1, self.w2),self.b2))
        # hidden_output_last = nn.ReLU(nn.Linear(nn.AddBias(hidden_output2,self.b2), self.w3))
        predicted_value = nn.AddBias(nn.Linear(hidden_output2, self.w_o), self.b_o)
        return predicted_value


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        current_accuracy = 0
        best_accuracy = 0
        epochs_total = 0
        patience = 5

       
        while current_accuracy <= 0.975:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                grad = nn.gradients(loss, [self.w1, self.b1, self.w_o, self.b_o, self.w2, self.b2])
                self.w1.update(grad[0], self.learning_rate)
                self.b1.update(grad[1], self.learning_rate)
                self.w_o.update(grad[2], self.learning_rate)
                self.b_o.update(grad[3], self.learning_rate)
                self.w2.update(grad[4], self.learning_rate)
                self.b2.update(grad[5], self.learning_rate)
            
            current_accuracy = dataset.get_validation_accuracy()

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                epochs_total = 0
            else:
                epochs_total += 1

   

            # Early stopping if no improvement for a certain number of epochs
            if epochs_total >= patience:
                epochs_total = 0
                break

        
             



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here

        hidden_layer_size = 512
        "*** YOUR CODE HERE ***"

        self.batch_size = 100
        self.learning_rate = -0.2

        self.w1 = nn.Parameter(self.num_chars, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
  

        self.w2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2 = nn.Parameter(1, hidden_layer_size)

        self.w1_hidden = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b1_hidden = nn.Parameter(1, hidden_layer_size)
    

        self.w2_hidden = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2_hidden = nn.Parameter(1, hidden_layer_size)


        self.w_o = nn.Parameter(hidden_layer_size, 5)
        self.b_o = nn.Parameter(1, 5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        for i in range(len(xs)):

            # if i == 0:
            #     # first letter
            #     hidden_output1= nn.ReLU(nn.AddBias(nn.Linear(xs[i], self.w1), self.b1))
            #     h = nn.AddBias(nn.Linear(hidden_output1, self.w2),self.b2)
            # else:
            #     hidden_output1= nn.ReLU(nn.AddBias(nn.Linear(xs[i], self.w1_hidden), self.b1_hidden))
            #     h = nn.ReLU(nn.AddBias(nn.Linear(hidden_output1, self.w2_hidden),self.b2_hidden))

            if i==0:
                Z1 = nn.AddBias(nn.Linear(xs[i],self.w1),self.b1)
                A1 = nn.ReLU(Z1)
                h = nn.AddBias(nn.Linear(A1,self.w2),self.b2)
            else:
                Z_one = nn.AddBias(nn.Add(nn.Linear(xs[i], self.w1), nn.Linear(h, self.w1_hidden)),self.b1_hidden)
                A_one = nn.ReLU(Z_one)
                Z_two = nn.AddBias(nn.Linear(A_one,self.w2_hidden),self.b2_hidden)
                h = nn.ReLU(Z_two)

        predicted_value = nn.AddBias(nn.Linear(h, self.w_o), self.b_o)
        return predicted_value

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        current_accuracy = 0
        best_accuracy = 0
        epochs_total = 0
        patience = 5

       
        while current_accuracy < 0.83:
            print("current accuracy", current_accuracy)
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                grad = nn.gradients(loss, [self.w1, self.b1, self.w_o, self.b_o, self.w2, self.b2, self.w1_hidden, self.b1_hidden, self.w2_hidden, self.b2_hidden])
                self.w1.update(grad[0], self.learning_rate)
                self.b1.update(grad[1], self.learning_rate)
                self.w_o.update(grad[2], self.learning_rate)
                self.b_o.update(grad[3], self.learning_rate)
                self.w2.update(grad[4], self.learning_rate)
                self.b2.update(grad[5], self.learning_rate)
                self.w1_hidden.update(grad[6], self.learning_rate)
                self.b1_hidden.update(grad[7], self.learning_rate)
                self.w2_hidden.update(grad[8], self.learning_rate)
                self.b2_hidden.update(grad[9], self.learning_rate)
            
            current_accuracy = dataset.get_validation_accuracy()

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                epochs_total = 0
            else:
                epochs_total += 1

            # Early stopping if no improvement for a certain number of epochs
            if epochs_total >= patience:
                break