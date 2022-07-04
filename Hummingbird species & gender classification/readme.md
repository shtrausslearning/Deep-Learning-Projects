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

```Python

''' Visualise Image Data '''
# Visualise a certain number of images in a folder using <code>ImageGrid</code>

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
