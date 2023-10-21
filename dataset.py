from lib import *
from make_data_path import make_data_path
from transform import DataTransform
from extract_anno_info import Anno_xml

class MyDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, ann_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = ann_xml
    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        
        return img, torch.tensor(gt)
    
    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        #get anno information
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path)
        
        #preprocessing
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:, 4])

        #BGR->RGB, (height, width, channels) -> (channels, height, width)
        img = torch.from_numpy(img[:,:, (2,1,0)]).permute(2, 0, 1)
        #ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
    def __len__(self):
        return len(self.img_list)


def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.tensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets



if __name__ == "__main__":
    root_dir = './data/VOCdevkit/VOC2012'
    train_img_path_list, val_img_path_list, train_anno_path_list, val_anno_path_list = make_data_path(root_dir)
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    color_mean = (104, 117, 123)
    input_size = 300
    anno_xml = Anno_xml(classes)
    transform = DataTransform(input_size, color_mean)
    train_dataset = MyDataset(train_img_path_list, train_anno_path_list, phase='train', transform=transform, ann_xml = anno_xml)
    val_dataset = MyDataset(val_img_path_list, val_anno_path_list, phase='val', transform=transform, ann_xml=anno_xml)

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)


    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }
    
    batch_iter = iter(dataloader_dict['train'])
    images, targets = next(batch_iter) #get a batch of data
    print(images.shape)
    print(len(targets))
    print(targets[0])