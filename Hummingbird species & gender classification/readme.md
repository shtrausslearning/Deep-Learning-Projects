![](https://i.imgur.com/icqghBp.png)

### 1 | Hummingbird monitoring

- The purpose of this project is to create an image <code>classifier</code> for hummingbird **species** and **genders** that visit feeders. 
- Such a classifier should be applicable to anywhere that hummingbirds migrate or breed, given additional datasets for those species.
- Hummingbird migration is otherwise reliant on individual bird watchers to see and report their observations. 
- If avid bird lovers setup a similar system, then conservation organizations would have better information on migratory and breeding patterns. 
- This knowledge can be used to determine if specific changes in the environment or ecology has positively or negatively impacted bird life.

### 2 | Reasons for turning to machine learning

- With the increased affordability of visual monitoring equipment, it has become practical for anyone to contribute to such a wonderful cause & help make each sighting more valid. 
- Often, due to <b>limitations of gear</b>, <b>poor photography/videography technique</b>, or **<span style='color:#B6DA32'>simply poor lighting conditions</span>**, it can be difficult for users, <b>let alone experts</b> to distinguish what specific bird was seen at the time of monitoring.
- In this study, we will focus our attention to a bird called the <b>hummingbird</b>. What is useful about this specie is that, despite being quite distinguishible in shape, <b>they have a variery of unique colouring</b>, which means that if images are of poor quality, it may be hard for humans or models to distinguish them.
- In the entire process of <b>expert identification</b> & dealing with various image related inconsistencies outlied previously, manual identification for monitoring can be quite labourous, so an automated system of identification can go a long way.
- In our quest to create an automated approach, we can be left with a collection or under or over exposed images that will create difficulties for the model to distinguish between different classes correctly. 

### 3 | Project goal 

- The ultimate goal is to have a classification system that can address such the above stated varieties in an image & correctly distinguish very similar bird species, it should be deployable at any feeder, which is important to the continued monitoring of hummingbird species and bird  migration patterns. 

### 4 | The dataset

- The dataset contains a main folder; <code>hummingbirds</code>, which contains image data split into <code>training</code>, <code>validation</code> & <code>test</code> sets, so the <code>train/test</code> split has already been done for us

```python
''' Folder Pathways'''
main_folder = '/kaggle/input/hummingbirds-at-my-feeders/'
train_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/train/'
val_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/valid/'
test_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/test/'
video_folder = '/kaggle/input/hummingbirds-at-my-feeders/video_test/'
```

```python

os.listdir(main_folder)

```

> ['video_test', 'All_images', 'hummingbirds']

```python
os.listdir(train_folder)
```

> ['Rufous_female', 'No_bird', 'Broadtailed_female', 'Broadtailed_male']

- Let's check how many images there are in each folder <code>Rufous female</code>, <code>No bird</code>, <code>Broadtailed female</code> & <code>Broadtailed male</code>
- We have a total of **4 classes**, so we'll be treating the problem as as <code>multiclass</code> classification problem

```python

class_types = len(os.listdir(train_folder))
print('Number of classes for Classification: ',class_types)
class_names = os.listdir(train_folder)
print(f'The class names are {class_names}\n')

print('Training dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(train_folder+i))))

print('\nValidation dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(val_folder+i))))
    
print('\nTest dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(test_folder+i))))
    
```

- The class distribution is even, the <code>training</code> data containing **100 samples** per class & both <code>validation</code> and <code>test</code> data containing **20 samples** per class

```
Number of classes for Classification:  4
The class names are ['Rufous_female', 'No_bird', 'Broadtailed_female', 'Broadtailed_male']

Training dataset:
Rufous_female:100
No_bird:100
Broadtailed_female:100
Broadtailed_male:100

Validation dataset:
Rufous_female:20
No_bird:20
Broadtailed_female:20
Broadtailed_male:20

Test dataset:
Rufous_female:20
No_bird:20
Broadtailed_female:20
Broadtailed_male:20
```

### 5 | Image Exploration

- Using the function <code>show_grid</code>, we can visualise the various types of class images
- We have four classes, let's make some remarks about each class 

```python

''' Visualise Image Data '''
# Visualise a certain number of images in a folder using ImageGrid

def show_grid(image_list,nrows,ncols,label_list=None,
              show_labels=False,savename=None,
              figsize=(20,10),showaxis='off'):
    
    if type(image_list) is not list:
        if(image_list.shape[-1]==1):
            image_list = [image_list[i,:,:,0] for i in range(image_list.shape[0])]
        elif(image_list.shape[-1]==3):
            image_list = [image_list[i,:,:,:] for i in range(image_list.shape[0])]
    fig = plt.figure(None, figsize,frameon=False)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     share_all=True)
    
    for i in range(nrows*ncols):
        ax = grid[i]
        img = Image.open(image_list[i])
        ax.imshow(img,cmap='Greys_r')  # The AxesGrid object work as a list of axes.
        ax.axis(showaxis)
        if show_labels:
            ax.set_title(class_mapping[y_int[i]])
    if savename != None:
        plt.savefig(savename,bbox_inches='tight')
        

```

#### **RUFOUS FEMALE**
- The addition of the <b>female as opposed to the male</b> is an interesting choice for the dataset, making it indeed very challenging for the model, due to the high similarity of different species' image values
- The female, unlike the male is very similar to the <b>broadtail female</b>, <b>especially when in the shade</b>, we actually have quite a few such cases, as seen in the images below
- However we can notice that in all images, colours of the back/rump are <b>quite dull, more saturated, but with hints of green</b>, just enough to be able to make out the green colour
- We can also note that the images have quite a <b>bit of "noise"</b>, that could potentally affect the accuracy of the model; <b>the feeder & the flag introduce red values into the images</b>, which can affect the model accuracy

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___10_0.png)

#### **BROADTAIL FEMALE**
- To the naked eye, there is <b>a lot of similarities</b> between the <b>Rufous</b> & <b>Broadtail Females</b>
- Without adequate lighting and refence to multiple frames (such as from a video), one could easily mislabel the species
- The boadtail female colour definitely stand out more, the two species so far have very similar bagrounds & the ocassional feeder

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___12_0.png)

#### **BROADTAIL MALE**
- Like the adult female, the males also have green and buffy flanks
- What separates the male broadtail from the female, and even from the Rufous female is the **distinctive rose/magenta throats**.
- It's quite likely the model would be easily able to classify any image containing the male from the rest
- We can clearly observe that the feeder, has both darker spots and lighter spots, lighter spots have values very similar values to the throat
- We can also note in some images, they don't have this distinctive red colour throat (at least to the naked eye), one possible reason being that the bird is <b>in the shade at the time of capture</b>. It's also possible that the images are not correctly labeled, which cannot also be ruled out. However we can clearly note that the model will need to adapt to images <b>taken under direct sunlight</b> and <b>different shade variations</b>, which create some problematic scenarios
- And <b>last but not least</b>, we can't rule out <b>immature males</b> from the pack as well. They are extremly similar to female as well

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___14_0.png)

#### **NO BIRD**
- Given the amount of background noise/clutter (non bird pixels) we have in our images, <b>no_bird images seem like significant additions</b>
- Especially important are the <b>flag</b> & <b>feeder</b> images which we saw in the background of some of the hummingbird images.
- No birds are present in any of the images, showing the environment around the feeder

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___16_0.png)

#### **REVIEWING THE DATA**
- In the context of bird monitoring, what I think this dataset outlines more than anything else is that:
  - You don't need to place cameras right next to the feeder (for some species can be offputting
  - The images don't need to be of perfect quality 

- Most hummingbirds are very similar in shape and are <b>mostly differentiable by their colours</b>:
  - So 1 channel CNN input network would be less effective, compared to a 3 channel network, and we have to rely on all colour channels to distinguish the species.
- Having gone through the images, we can see that the current dataset is quite a challenging one. A lot of other hummingbirds, especially male have very <b>identifiable feather colours</b>, however in this dataset, aside from the <b>broadtail male</b>, broadtail and rufus female hummingbirds <b>seem amost identical to the naked eye</b>.
- The <b>model heavily relies on accurate initial training label data</b>

