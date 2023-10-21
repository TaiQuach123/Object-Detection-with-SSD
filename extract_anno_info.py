from lib import *
from make_data_path import make_data_path


class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, xml_path):
        ret = []

        xml = ET.parse(xml_path).getroot()
        img_size = xml.find('size')
        height, width = int(img_size.find('height').text), int(img_size.find('width').text)


        for obj in xml.iter('object'):
            if int(obj.find('difficult').text) == 1:
                continue
            
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1
                if pt == 'xmin' or pt == 'xmax':
                    pixel /= width
                else:
                    pixel /= height

                bndbox.append(pixel)
            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]

        return np.array(ret)


if __name__ == "__main__":
    root_dir = './data/VOCdevkit/VOC2012'
    train_img_path_list, val_img_path_list, train_anno_path_list, val_anno_path_list = make_data_path(root_dir)
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


    
    xml_path = val_anno_path_list[1]
    anno_xml = Anno_xml(classes)
    print(anno_xml(xml_path))