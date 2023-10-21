from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords,\
    ToPercentCoords, Resize, SubtractMeans

from make_data_path import make_data_path
from extract_anno_info import Anno_xml
from lib import *

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(), #convert image from int to float 32
                ToAbsoluteCoords(), #back annotation to normal type
                PhotometricDistort(), #change color by random 
                Expand(color_mean),
                #RandomSampleCrop(), #randomcrop image
                RandomMirror(), #xoay anh nguoc lai
                ToPercentCoords(), #chuan hoa annotation data ve [0-1]
                Resize(input_size),
                SubtractMeans(color_mean),

                ]),  
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    root_dir = './data/VOCdevkit/VOC2012'
    train_img_path_list, val_img_path_list, train_anno_path_list, val_anno_path_list = make_data_path(root_dir)
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]



    #read img
    img_file_path = train_img_path_list[0]
    img = cv2.imread(img_file_path) #BGR
    height, width, channels = img.shape

    #annotation information

    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(train_anno_path_list[0])
    #print(anno_info_list[:, :4], anno_info_list[:, -1])
    #plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


    #prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    #transform train img
    phase = 'train'
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

    #transform val img
    phase = 'val'
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, -1])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

