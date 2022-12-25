import random
import os
from keras.preprocessing import image
from keras.utils import np_utils

''' 

Load Images into Numpy Format 

1. select directory (directory)
2. Folder organisation /Directory/(class id)/(class imgs)
   
   eg. Directory/0/1.jpg
       Directory/2/2.jpg
       Directory/1/3.jpg
       
3. Image data stored in img_data
4. Shuffle Dataset
5. One-Hot Encode Labels

Returns X_train,Y_train

'''

def create_imgdata(directory,img_size=(224,224)):

    folders = os.listdir(directory)
    print('[note] class folders')
    print(folders)

    img_data = []; labels = []; ii=0
    for folder in folders:
        path = os.path.join("Train", folder)
        print(path, ii) 
        for im in os.listdir(path):
            try:
                img = image.load_img(os.path.join(path,im),
                                     target_size = img_size)
                img_array = image.img_to_array(img)
                img_data.append(img_array)
                labels.append(ii)
            except:
                pass
        ii+=1

    ''' Shuffle Dataset '''
    # using the random module

    combined_dataset = list(zip(img_data, labels))
    random.shuffle(combined_dataset)
    image_data[:], labels[:] = zip(*combined_dataset)

    ''' OHE the labels '''
    # Store training data in numpy array & OHE labels
    # using the keras utility; np_utlis

    X_train = np.array(img_data)
    Y_train = np.array(labels)
    Y_train = np_utils.to_categorical(Y_train)
    
    return X_train,Y_train
