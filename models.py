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
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score_node = self.run(x)
        score = nn.as_scalar(score_node)

        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        converged = False
        while not converged:
            misclassified = False
            for x, y in dataset.iterate_once(batch_size):
                y_scalar = nn.as_scalar(y)
                prediction = self.get_prediction(x)
                if abs(prediction-y_scalar)>1e-6:  
                    misclassified = True
                    # Update weights
                    direction = x
                    self.w.update(direction, nn.as_scalar(y))
            if not misclassified:
                converged = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(1, 64)  # Parameters for the first layer
        self.b1 = nn.Parameter(1, 64)  # Bias for the first layer
        self.W2 = nn.Parameter(64, 1)  # Parameters for the second layer
        self.b2 = nn.Parameter(1, 1)   # Bias for the second layer
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Forward pass through the network
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        return x

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
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Set hyperparameters
        learning_rate = 0.01
        num_epochs = 1000
        batch_size = 1
        
        total_examples = sum(1 for _ in dataset.iterate_once(batch_size))
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Forward pass
                loss = self.get_loss(x_batch, y_batch)
                total_loss += nn.as_scalar(loss)
                # Backward pass
                grads = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                # Update parameters
                self.W1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.W2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)
            # Print average loss for the epoch
            avg_loss = total_loss / total_examples
            print("Epoch {}, Average Loss: {:.6f}".format(epoch+1, avg_loss))
            if avg_loss < 0.02:
                break

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
        self.W1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.W2 = nn.Parameter(256, 10)
        self.b2 = nn.Parameter(1, 10)

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
        # Forward pass
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        return x

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
        
        scores = self.run(x)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.1
        num_epochs = 20
        batch_size = 60 

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Forward pass
                loss = self.get_loss(x_batch, y_batch)
                total_loss += nn.as_scalar(loss)
                # Backward pass
                grads = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                # Update parameters
                self.W1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.W2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)
            val_accuracy = dataset.get_validation_accuracy()
            print("Epoch {}, Avg. Loss: {:.4f}, Validation Accuracy: {:.2f}%".format(epoch+1, total_loss, val_accuracy * 100))
            #  higher stopping threshold
            if val_accuracy >= 0.975:
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
        "*** YOUR CODE HERE ***"

        hidden_size = 250
        
        self.W_initial = nn.Parameter(self.num_chars, hidden_size)
        self.b_initial = nn.Parameter(1, hidden_size) 
        self.W_hidden = nn.Parameter(hidden_size, hidden_size)
        self.b_hidden = nn.Parameter(1, hidden_size)

        self.W_output = nn.Parameter(hidden_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))
        
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
    
        # Initialize hidden state
        h = nn.Linear(xs[0], self.W_initial)
        h = nn.AddBias(h, self.b_initial)
        h = nn.ReLU(h)

        # Update hidden state for each character
        for x in xs[1:]:
            z = nn.Linear(x, self.W_initial)
            z = nn.AddBias(z, self.b_initial)
            h = nn.ReLU(nn.Add(nn.Linear(h, self.W_hidden), nn.Linear(z, self.W_hidden)))

        # Output layer
        output = nn.Linear(h, self.W_output)
        output = nn.AddBias(output, self.b_output)
        return output

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
        scores = self.run(xs)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
            
        num_epochs = 20
        batch_size = 32
        learning_rate= 0.1

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Forward pass
                loss = self.get_loss(x_batch, y_batch)
                total_loss += nn.as_scalar(loss)
                # Backward pass
                grads = nn.gradients(loss, [self.W_initial, self.b_initial, self.W_hidden, self.b_hidden, self.W_output, self.b_output])
                # Update parameters
                for param, grad in zip([self.W_initial, self.b_initial, self.W_hidden, self.b_hidden, self.W_output, self.b_output], grads):
                    param.update(grad, -learning_rate)
            print("Epoch {}, Avg. Loss: {:.4f}".format(epoch+1, total_loss))

