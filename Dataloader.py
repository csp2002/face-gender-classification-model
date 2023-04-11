from PIL import Image
import numpy as np
class DataLoader:
    images = []
    labels = []
   
    dataPath = 'face/'

    train_images = []
    train_labels = []
   
    
    validation_images = []
    validation_labels = []
  
    test_images = []
    test_labels = []
    dataCount=12000

    
#加载人脸数据
    def loadfaceData(self):
        nameHandle1=open('male_names.txt','r')
        nameHandle2=open('female_names.txt','r')
        male_name=nameHandle1.readlines()
        female_name=nameHandle2.readlines()
        num1=len(male_name)
        num2=len(female_name)
        
        for i in range(num1):
            male_name1=male_name[i].replace("\n","")
            picpath1=self.dataPath+str(male_name1)
            image1=self.loadPicArray(picpath1)
            label1=1
            self.images.append(image1)
            self.labels.append(label1)
            
        for j in range(num2):
            female_name1=female_name[j].replace("\n","")
            picpath2=self.dataPath+str(female_name1)
            image2=self.loadPicArray(picpath2)
            label2=0
            self.images.append(image2)
            self.labels.append(label2)
            
        nameHandle1.close()
        nameHandle2.close()
        #打乱数据，使用相同的次序打乱images和labels，保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        
        

        #按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 0.40)
        validationIndex = int(self.dataCount * 0.50)
        self.train_images = self.images[0 : trainIndex]
        self.train_labels = self.labels[0 : trainIndex]
        
        self.validation_images = self.images[trainIndex : validationIndex]
        self.validation_labels = self.labels[trainIndex : validationIndex]
        
        self.test_images = self.images[validationIndex : ]
        self.test_labels = self.labels[validationIndex : ]
        


    #读取图片数据，得到图片对应的像素值的数组，均一化到0-1之前
    def loadPicArray(self, picFilePath):
        picData = Image.open(picFilePath)
        picData_L=picData.convert("L")
        picArray = np.array(picData_L).flatten() / 255.0
        return picArray

    def getTrainData(self):
        return self.train_images, self.train_labels
    def getValidationData(self):
        return self.validation_images, self.validation_labels

    def getTestData(self):
        return self.test_images, self.test_labels

