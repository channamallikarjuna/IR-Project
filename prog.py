from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense ,Dropout,Flatten

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from PIL import Image

from os.path import dirname
import warnings
warnings.filterwarnings("ignore")





base_model = VGG19(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
model.summary()

def extract( img):
    # Resize the image
    img = img.resize((224, 224))
    # Convert the image color space
    img = img.convert('RGB')
    # Reformat the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract Features
    feature = model.predict(x)[0]
    return feature / np.linalg.norm(feature)   
      	
"""# Iterate through images (Change the path based on your image location)
print(os.getcwd())
for folder in os.listdir(os.getcwd()+"/dataset/training_set"):
    #print(folder)
    for img_path in os.listdir(os.getcwd()+"/dataset/training_set/"+folder):
        if img_path.endswith('.jpg'):
            #print(os.getcwd()+"/dataset/training_set/"+folder+"/"+img_path)
            # Extract Features
            feature = extract(img=Image.open(os.getcwd()+"/dataset/training_set/"+folder+"/"+img_path))
            # Save the Numpy array (.npy) on designated path
            feature_path = os.getcwd()+"/dataset/training_set/"+folder+"/"+os.path.splitext(img_path)[0]+".npy"
            np.save(feature_path, feature)"""




# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
features = []
img_paths = []
for folder in os.listdir(os.getcwd()+"/dataset/training_set"):
    #print(folder)
    for img_path in os.listdir(os.getcwd()+"/dataset/training_set/"+folder):
        if img_path.endswith('.jpg'):
            features.append(np.load(os.getcwd()+"/dataset/training_set/"+folder+"/"+os.path.splitext(img_path)[0]+".npy"))
            img_paths.append(os.getcwd()+"/dataset/training_set/"+folder+"/"+img_path)

features = np.array(features)        



#print(img_paths)
"""# Insert the image query
img = Image.open(os.getcwd()+"/604.jpg")
object="flowers"
count=0
# Extract its features
query = extract(img)"""
great_sm=0
for folder in os.listdir(os.getcwd()+"/dataset/test_set"):
    print("_______________________________________________________")
    print("\n" + folder)
    avg=0
    s=0
    for img_path in os.listdir(os.getcwd()+"/dataset/test_set/"+folder):

        if img_path.endswith('.jpg'):
            #print(os.getcwd()+"/dataset/test_set/"+folder+"/"+img_path)
            # Extract Features
            query = extract(img=Image.open(os.getcwd()+"/dataset/test_set/"+folder+"/"+img_path))
            object=folder
            count=0


            #print(query)

            # Calculate the similarity (distance) between images
            dists = np.linalg.norm(features - query, axis=1)
            # Extract 30 images that have lowest distance
            ids = np.argsort(dists)[:30]

            scores = [(dists[id], img_paths[id]) for id in ids]
            # Visualize the result
            axes=[]
            fig=plt.figure(figsize=(8,8))
            for a in range(5*6):
                score = scores[a]
                axes.append(fig.add_subplot(5, 6, a+1))
                subplot_title=str(score[0])+" "+os.path.basename(dirname(score[1]))
                #print(os.path.splitext(score[1])[0])
                k=os.path.basename(dirname(score[1]))
                if(k==object):
                    count=count+1

                axes[-1].set_title(subplot_title)  
                plt.axis('off')
                plt.imshow(Image.open(score[1]))
            fig.tight_layout()
            #plt.show()
            precision=count/30
            s=s+precision
            #recall=count/len(img_path)
            print(folder + "   precision = "+ str(precision))
            #print("Recall = "+ str(recall))

    avg=s/10
    print("Average precision of  " + folder + "  = "+str(avg))
    great_sm=great_sm+avg
print("Average precision = "+str(great_sm/10))



