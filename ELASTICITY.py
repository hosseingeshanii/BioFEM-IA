import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.initializers import Constant





# Global variables for model and optimizer
model = None
optimizer = None

class CustomMLP(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_value=0.5):
        super(CustomMLP, self).__init__()

        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu', 
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu', 
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))
        self.fc3 = tf.keras.layers.Dense(1, activation='linear', 
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))  # Output layer
        
        self.output_value = output_value  # Desired output value


    def build(self, input_shape):
        super(CustomMLP, self).build(input_shape)
        
        
        # Initialize biases to desired output value
        self.fc1.bias.assign(tf.zeros_like(self.fc1.bias))
        self.fc2.bias.assign(tf.zeros_like(self.fc2.bias))
        self.fc3.bias.assign(tf.constant(self.output_value, shape=self.fc3.bias.shape))


    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        E = self.fc3(x)
        return E

def initialize_model(output_value=0.5):
    global model, optimizer
    input_size = 2
    hidden_size = 5

    model = CustomMLP(input_size, hidden_size, output_value)
    model.build((None, input_size))  # Build the model to initialize weights
    #optimizer = tf.keras.optimizers.SGD(learning_rate=10.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 


def load_model(epoch):
    # Construct the model filename based on the epoch number
    model_filename = f"model_epoch_{epoch}"
    
    try:
        # Load the model from the specified file
        loaded_model = tf.keras.models.load_model(model_filename)
        print(f"Model loaded from {model_filename}")
        return loaded_model
    except Exception as e:
        print(f"Error loading model from {model_filename}: {e}")
        return None
    


def start_up(init_flag, epoch=None, learning_rate=0.001, output_value=0.5):
    global model
    global optimizer  # Ensure optimizer is recognized as a global variable

    if init_flag:
        # Initialize the model
        initialize_model(output_value)

        # Initialize the optimizer with the provided learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        print(f"Model and optimizer initialized with learning rate: {learning_rate}.")
    else:
        # Load model and optimizer from saved state
        if epoch is not None:
            model = load_model(epoch)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Reinitialize optimizer
            if model is not None:
                print(f"Model loaded from epoch {epoch} with learning rate: {learning_rate}.")
            else:
                print("Failed to load model.")
        else:
            print("Epoch number is required to load the model.")

def compute_E(x1, x2):
    global model
    inputs = np.array([[x1, x2]], dtype=np.float32)
    E = model(inputs).numpy().item()
    return E

def update_learning_rate(new_lr):
    global optimizer
    optimizer.learning_rate = new_lr
    print(f"Learning rate updated to: {new_lr}")
    
def update_weights(inputs_list, R_list, dR_dE_numerical, epoch, save_flag=False):
    global model
    global optimizer
    inputs = np.array(inputs_list, dtype=np.float32)
    R_true = np.array(R_list, dtype=np.float32).reshape(-1, 1)  # Use R_list as R_true
    #print(R_true)
    if len(inputs.shape) == 1:
        inputs = np.expand_dims(inputs, axis=0)

    # Compute loss directly using R_list
    loss = tf.reduce_mean((R_true - 0) ** 2)  # Assuming R_true should be zero

    # Use chain rule to compute the gradients of loss with respect to E
    grad_loss_wrt_R = 2 * (R_true - 0) / R_true.size  # dLoss/dR, assuming R_true should be zero
    grad_R_wrt_E = dR_dE_numerical
    grad_loss_wrt_E = grad_loss_wrt_R[0] * grad_R_wrt_E  # dLoss/dR * dR/dE

    # Save current weights for comparison

    # Manually calculate gradients with respect to trainable variables
    with tf.GradientTape() as tape:
        E = model(inputs)
    model_gradients = tape.gradient(E, model.trainable_variables, output_gradients=tf.convert_to_tensor(np.array([grad_loss_wrt_E], dtype=np.float32).reshape(-1, 1)))
        
        
    # Apply gradients
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))

    
    if save_flag:
        model_save_path = f"model_epoch_{epoch}"
        model.save(model_save_path, save_format='tf')
        print(f"Model saved at {model_save_path}")

