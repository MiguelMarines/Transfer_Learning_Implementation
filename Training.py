# ====================================================================================================================================== #
#                                                             TRANSFER LEARNING                                                          #
# ====================================================================================================================================== #
# Author: Miguel Marines


# ====================================================================================================================================== #
#                                                               LIBRARIES                                                                #
# ====================================================================================================================================== #
# DEEP LEARNING ------------------------------------------------------------------------------------------------------------------------
import torch                                # Provides functionalities for deep learning and numerical computing.
import torch.nn as nn                       # Provides modules and classes to create and train neural networks.
import torch.nn.functional as F             # Provides functions used in neural network operations and loss computations.
from torch.optim import Adam                # Provides optimization algorithms, in this case the Adam optimiser.


# DEEP LEARNING AND VISION -------------------------------------------------------------------------------------------------------------
import torchvision.transforms as T          # Provides common image transformations.
from torchvision import models              # Provides pre-trained models for computer vision tasks.


# DATA ---------------------------------------------------------------------------------------------------------------------------------
import torchvision.datasets as datasets     # Provides the dataset and creates dataset objects (with our own objects).

from torch.utils.data import DataLoader     # Converts the dataset to an iterable object to make the minibatches automatically.
from torch.utils.data import sampler        # Used to create random samples from the data.


# DIRECTORIES AND FILES ----------------------------------------------------------------------------------------------------------------
from os.path import join                    # Used to join one or more path components, taking into account the operating system's path separator.
import pathlib                              # Provides functionalities to manipulate paths, access file data, iterate over directories, and perform file operations.
import glob                                 # Provides functions for searching file patterns or matching filenames.


# GRAPHS -------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt             # Provides a collection of functions for creating visualizations, such as plots, charts, histograms, etc.





# ====================================================================================================================================== #
#                                                           SELECT DEVICE TO USE                                                         #
# ====================================================================================================================================== #
# Select whether to use GPU or CPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Print device to use.
print("\nDevice:", str(device))





# ====================================================================================================================================== #
#                                                      TRANSFORM AND NORMALIZE IMAGES                                                    #
# ====================================================================================================================================== #
# Transform images to tensors and normalize them. 
transformer =   T.Compose([
                T.Resize((224, 224)),                         # Resize the input image to the given size (H, W).
                T.ToTensor(),                                 # Convert image to tensor (Tensor: Multi-dimensional matrix containing elements of a single data type).
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize a tensor image with mean and standard deviation.
            ])





# ====================================================================================================================================== #
#                                                                   LOAD DATA                                                            #
# ====================================================================================================================================== #
# Training Path
Train_Path = "/Users/.../seg_train"

# Testing Path
Test_Path = "/Users/.../seg_test"


# Load train data and shuffle it.
train_loader = DataLoader(datasets.ImageFolder(Train_Path, transform = transformer), batch_size = 32, shuffle = True)

# Load test data and shuffle it.
test_loader = DataLoader(datasets.ImageFolder(Test_Path, transform = transformer), batch_size = 311, shuffle = True)





# ====================================================================================================================================== #
#                                                     GET CATEGORIES AND CLASSES                                                         #
# ====================================================================================================================================== #
# Get categories and classes.
root = pathlib.Path(Train_Path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Remove .DS_Store in MAC from classes.
classes.pop(0)

# Print classes.
print("\n\nClases: ", end="")
print(', '.join(classes))





# ====================================================================================================================================== #
#                                                 GET NUMBER OF TRAINING AND TESTING IMAGES                                              #
# ====================================================================================================================================== #
# Number of the training and testing images.
train_count = len(glob.glob(Train_Path + '/**/*.jpeg'))
test_count = len(glob.glob(Test_Path + '/**/*.jpeg'))

# Print number of the training and testing images.
print("\n\nTraining Images: " + str(train_count))
print("Testing Images: " + str(test_count))
print("\n\n")





# ====================================================================================================================================== #
#                                                               ACCURACY                                                                 #
# ====================================================================================================================================== #
# Accuracy function with parameters model(ResNet 34) and loader(batch).
def accuracy(model, loader):
    
    # Variables to keep track of the number of correctly predicted samples and the total number of samples.
    num_correct = 0
    num_total = 0

    # Set the model to evaluation mode.
    model.eval()

    # The model's computation is performed on the specified device.
    model = model.to(device = device)

    # Disables gradient computation.
    with torch.no_grad():
        
        # Loop that provides batches of data.
        for (xi, yi) in loader:

            # Obtain images(tensors) and convert to the appropriate data type.
            xi = xi.to(device = device, dtype = torch.float32)

            # Obtain classes(tensors) and convert to the appropriate data type.
            yi = yi.to(device = device, dtype = torch.long)
            
            # Predict class labels by finding the index of the maximum score.
            scores = model(xi)
            _, pred = scores.max(dim = 1)

            # Count the correct predictions.
            num_correct += (pred == yi).sum()

            # Count total predictions.
            num_total += pred.size(0)

        # Compute and return accuracy.
        return float(num_correct)/num_total





# ====================================================================================================================================== #
#                                                           LOAD PRE-TRAINED MODEL                                                       #
# ====================================================================================================================================== #
# ResNet model.
model_resnet34 = models.resnet34(weights = True)


# ADJUST MODEL -------------------------------------------------------------------------------------------------------------------------
model_aux = nn.Sequential(*list(model_resnet34.children()))

# Eliminate last layer.
model_aux = nn.Sequential(*list(model_resnet34.children())[:-1])

# Freeze these parameters and keep them fixed during training.
for i, parameter in enumerate(model_aux.parameters()):
    parameter.requires_grad = False

# Print parameters.
# for i, parameter in enumerate(model_aux.parameters()):
#     print(i, parameter.requires_grad)

# Print model.
# print(model_resnet34)


# Build a neural network model by sequentially stacking and chaining multiple layers.
model1 = nn.Sequential(model_aux,
                       nn.Flatten(), 
                       nn.Linear(in_features = 512, out_features = 7, bias = True))





# ====================================================================================================================================== #
#                                                           TRAINING LOOP                                                                #
# ====================================================================================================================================== #
# Train function with parameters model(ResNet 34), optimizer, and number of epochs.
def train(model, optimiser, epochs):

    # The model's computation is performed on the specified device.
    model = model.to(device = device)

    # Variable to check best accuracy.
    best_accuracy = 0.0
    
    # Arrays to store the values for the plot.
    cost_values = []
    acc_values = []

    # Loop to iterate according to the number of epochs.
    for epoch in range(epochs):

        # Loop that provides batches of data.
        for i, (xi, yi) in enumerate(train_loader):

            # Set the model to train mode.
            model.train()
            
            # Obtain images(tensors) and convert to the appropriate data type.
            xi = xi.to(device = device, dtype = torch.float32)
            
            # Obtain classes(tensors) and convert to the appropriate data type.
            yi = yi.to(device = device, dtype = torch.long)
            
            # Predict class labels by finding the index of the maximum score.
            scores = model(xi)

            # Calculate the cost or loss using the cross-entropy loss function.
            cost = F.cross_entropy(input = scores, target = yi)
        
            # Optimizer object (adam optimiser) that is responsible for updating the parameters of the model based on the computed gradients.
            # Set the gradients of all the model parameters to zero.
            optimiser.zero_grad()

            # Backpropagation to compute the gradients of the model parameters with respect to the cost.
            cost.backward()

            # Apply the computed gradients to update the parameters of the model.
            optimiser.step()
            
        # Compute accuracy with the accuracy function.
        acc = accuracy(model, train_loader)

        # Check best accuracy.
        if(acc > best_accuracy):
            best_accuracy = acc
            
            # Save Best Model ----------------------------------------------------------------------------------------------------------
            torch.save(model1.state_dict(), '/Users/.../resnet34_modified.model')

        # Store the cost and accuracy values in arrays for plotting.
        cost_values.append(cost.item())
        acc_values.append(acc)

        # Print the results of the epoch.
        print(f'Epoch: {epoch}, Cost: {cost.item()}, Accuracy: {acc}')
    
    # Print best accuracy.
    print("\nBest Training Accuracy: " + str(best_accuracy))

    # Plot Cost
    plt.plot(range(epochs), cost_values)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost vs Epoch')
    plt.show()

    # Plot Accuracy
    plt.plot(range(epochs), acc_values)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.show()





# ====================================================================================================================================== #
#                                                       TRAINIG, TESTING, ADN RESULTS                                                    #
# ====================================================================================================================================== #
# Learning Rate = 0.0005
lr = 5e-4

# Number of Epochs
epochs = 20

# Adam Optimizer
optimiser = Adam(model1.parameters(), lr = lr, betas = (0.9, 0.999))

# Train 
train(model1, optimiser, epochs)

# Test
print("\n\nTesting Accuracy: " + str(accuracy(model1, test_loader)))

# Save Best Model -----------------------------------------------------------------------------------------------------------------
# torch.save(model1.state_dict(), '/Users/.../resnet34_modified.model')

# Print Model
# print(model1.state_dict())