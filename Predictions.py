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


# DEEP LEARNING AND VISION -------------------------------------------------------------------------------------------------------------
import torchvision.transforms as T          # Provides common image transformations.
from torchvision import models              # Provides pre-trained models for computer vision tasks.


# DIRECTORIES AND FILES ----------------------------------------------------------------------------------------------------------------
from os.path import join                    # Used to join one or more path components, taking into account the operating system's path separator.
from io import open                         # Used to open files for reading or writing.
import pathlib                              # Provides functionalities to manipulate paths, access file data, iterate over directories, and perform file operations.
import glob                                 # Provides functions for searching file patterns or matching filenames.


# IMAGES -------------------------------------------------------------------------------------------------------------------------------
from PIL import Image                       # Provides methods for opening, manipulating, and saving images in various formats.


# TENSORS ------------------------------------------------------------------------------------------------------------------------------
from torch.autograd import Variable         # Provides a way to wrap a tensor and record operations performed on it.

# TIME ---------------------------------------------------------------------------------------------------------------------------------
import time                                 # Used to manage time.





# ====================================================================================================================================== #
#                                                           SELECT DEVICE TO USE                                                         #
# ====================================================================================================================================== #
# Select whether to use GPU or CPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Print device to use.
# print("\nDevice:", str(device))





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

# Prediction Path
Prediction_Path = "/Users/.../seg_pred"





# ====================================================================================================================================== #
#                                                     GET CATEGORIES AND CLASSES                                                         #
# ====================================================================================================================================== #
# Get categories and classes.
root = pathlib.Path(Train_Path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Remove .DS_Store in MAC from classes.
classes.pop(0)

# Print classes.
# print("\n\nClases: ", end="")
# print(', '.join(classes))





# ====================================================================================================================================== #
#                                                           LOAD PRE-TRAINED MODEL                                                       #
# ====================================================================================================================================== #
model_resnet34 = models.resnet34(weights = True)

# ADJUST MODEL -------------------------------------------------------------------------------------------------------------------------
model_aux = nn.Sequential(*list(model_resnet34.children()))

# Eliminate last layer.
model_aux = nn.Sequential(*list(model_resnet34.children())[:-1])

model1 = nn.Sequential(model_aux,
                       nn.Flatten(), 
                       nn.Linear(in_features = 512, out_features = 7, bias = True))





# ====================================================================================================================================== #
#                                                            LOAD TRAINED MODEL                                                          #
# ====================================================================================================================================== #
# Path of the best model obtained from the training.
checkpoint = torch.load('/Users/.../resnet34_modified.model')

# Load best model from the training and testing.
model1.load_state_dict(checkpoint)               # Feed the checkpoint inside the model.
model1.eval()                                    # Evaluation mode to set droput and batch normalization to evaluation mode and get consisten results.





# ====================================================================================================================================== #
#                                                            PREDICTION FUNCTION                                                         #
# ====================================================================================================================================== #
def prediction_function(img_path, transformer):
    
    #  Opens image file located at the specified path and creates an image object.
    image = Image.open(img_path)
    
    #  Transform the image into a tensor.
    image_tensor = transformer(image).float()
    
    # Add an extra dimension to the image tensor.
    image_tensor = image_tensor.unsqueeze_(0)
    
    # If devise GPU is being used.
    if torch.cuda.is_available():
        image_tensor.cuda()
    
    # Input with the image in tensor format to make the prediction.
    input = Variable(image_tensor)
    
    # Output of the prediction.
    output = model1(input)
    
    # Extracts the predicted class index from the output tensor.
    index = output.data.numpy().argmax()
    
    # Get the predicted class with the index of a predefined list or array of classes.
    pred = classes[index]

    # Assigning corresponding units of help according to the problematic situation.
    if(pred == "Altercation"):
        pred = "Altercation => Police Departement"
    if(pred == "Robbery"):
        pred = "Robbery => Police Departement"
    if(pred == "Accident"):
        pred = "Accident => Towing Unit, Ambulance and Transit Police"
    if(pred == "Fire"):
        pred = "Fire => Firefighters and Transit Police"
    if(pred == "Infrastructure Damage"):
        pred = "Infrastructure Damage => Repair Unit, Transit Police"
    if(pred == "Protest"):
        pred = "Protest => Police Force and Transit Police"
    if(pred == "Water"):
        pred = "Flood Risk => Drainage Department and Transit Police"
    
    # Return prediction.
    return pred





# ====================================================================================================================================== #
#                                                        PREDICTIONS AND RESULTS                                                         #
# ====================================================================================================================================== #
# Variable to store the system sign.
sign = """
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                               ║
║                                                        ASSISTANCE SITUATIONS                                                  ║
║                                                            CONTROL SYSTEM                                                     ║
║                                                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
# Print Sign
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
print(sign)

# Cycle to keep running the predictions.
for index in range(30):
    
    # Variable to store the common path for the images.
    images_path = glob.glob(Prediction_Path + '/*.jpeg')

    # Dictionary to store predictions.(Image name key and prediction)
    pred_dict = {}

    # Save in dictionary image name key and prediction.
    for i in images_path:
        pred_dict[i[i.rfind('/') + 1:]] = prediction_function(i, transformer)

    # Dictionary to store predictions with coordinates without image format.(Image name key and prediction)
    pred_coordinates_dict = {}

    # Remove .jpeg just to leave coordinates.
    for key, value in pred_dict.items():
        new_key = key.replace('.jpeg', '')
        pred_coordinates_dict[new_key] = value


    # Print dictionary with predictions.
    for name in pred_coordinates_dict.keys():
        # Convert elements to string.
        prediction = [str(pred_coordinates_dict[name])]
        # Print results.
        if(index < 10):
            print(f"ID: 000{index}     Coordinates: {name}     Units of Assistance: {prediction}")
        if(index >= 10):
            print(f"ID: 00{index}     Coordinates: {name}     Units of Assistance: {prediction}")
        print("\n")

    time.sleep(5)

