from lib import *



def make_data_path(root_dir):
    img_path_template = os.path.join(root_dir, 'JPEGImages', '%s.jpg')
    annotation_path_template = os.path.join(root_dir,'Annotations', '%s.xml')

    train_img_path_list = []
    val_img_path_list = []
    train_anno_path_list = []
    val_anno_path_list = []


    train_id_names = os.path.join(root_dir, 'ImageSets', 'Main', 'train.txt')
    val_id_names = os.path.join(root_dir, 'ImageSets', 'Main', 'val.txt')

    for line in open(train_id_names, 'r'):
        line = line.strip()
        img_path = (img_path_template % line)
        anno_path = (annotation_path_template % line)

        train_img_path_list.append(img_path)
        train_anno_path_list.append(anno_path)

    for line in open(val_id_names, 'r'):
        line = line.strip()
        img_path = (img_path_template % line)
        anno_path = (annotation_path_template % line)

        val_img_path_list.append(img_path)
        val_anno_path_list.append(anno_path)
    




    return train_img_path_list, val_img_path_list, train_anno_path_list, val_anno_path_list


if __name__ == "__main__":

    root_dir = './data/VOCdevkit/VOC2012'
    train_img_path_list, val_img_path_list, train_anno_path_list, val_anno_path_list = make_data_path(root_dir)

    print(train_img_path_list[0])
    print(train_anno_path_list[0])