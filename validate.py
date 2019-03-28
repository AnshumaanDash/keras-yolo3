import numpy as np
import keras.backend as K
import colorsys
import os
from PIL import Image, ImageFont, ImageDraw
from keras.layers import Input, Lambda
from keras.models import load_model
from yolo3.utils import get_random_data
from yolo3.model import preprocess_true_boxes
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from tqdm import tqdm

class Validate():

    def __init__(self):

        self.annotation_path = '2007_train.txt'
        self.output_dir = 'logs/000/output/'
        self.classes_path = 'model_data/voc_classes.txt'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.model_path = 'model_data/ep069-loss18.414-val_loss20.897.h5'
        self.class_names = self.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = self.get_anchors(self.anchors_path)
        self.score = 0.3
        self.iou = 0.45
        self.gpu_num = 1

        self.input_shape = (416,416) # multiple of 32, hw
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        
    def detect_all(self):
        
        detection_results = []
        detection_labels = np.array([0]*self.num_classes) 
        data_get = self.get_generator()
        
        for i, (image, true_boxes, true_labels) in tqdm(enumerate(data_get)):
            pred_boxes, conf, pred_labels = self.detect(image)
            
            sorted_inds = np.argsort(-conf)
            repeat_mask = [True]*len(true_boxes)
            matched_labels = []
            global_index = np.arange(len(true_labels))
            true_boxes = np.array(true_boxes)
            true_labels = np.array(true_labels)

            image_results = []
            image_labels = [0]*self.num_classes

            for tl in true_labels:
                image_labels[tl] += 1


            for i in sorted_inds:

                label_mask = (pred_labels[i] == true_labels)
                #print(f'label mask: {label_mask}')
                #print(f'repeat mask: {repeat_mask}')
                #print(f'& operationn result: {(repeat_mask)&(label_mask)}')
                index_subset = global_index[(repeat_mask)&(label_mask)]
                #print(f'index subset: {index_subset}')
                true_boxes_subset = true_boxes[(repeat_mask)&(label_mask)]
                #############################################################
                #print(f'true box subset: {true_boxes_subset}')
                #print(f'pred box : {pred_boxes}')
                idx = self._find_detection(pred_boxes[i], true_boxes_subset, index_subset)

                if idx != -1: 
                    matched_labels.append(idx)
                    repeat_mask[idx] = False

                image_results.append([pred_labels[i], conf[i], 1 if idx != -1 else 0])
            
            detection_results.extend(image_results)
            detection_labels += np.array(image_labels)
        
        
        detection_results = np.array(detection_results)

        ap_dic = {}
        for class_ind, num_gts in enumerate(detection_labels):
            
            class_detections = detection_results[detection_results[:,0]==class_ind]
            
            ap = self.compute_ap(class_detections, num_gts)

            ap_dic[self.class_names[class_ind]] = ap
            
        
        _AP_items = [[class_label, ap] for class_label, ap in ap_dic.items()]
        AP_items = sorted(_AP_items, key=lambda x: x[1], reverse=True)

        for class_label, ap in AP_items:
            print("AP( %s ): %.3f"%(class_label, ap))

        print('-------------------------------')
        print("mAP: %.3f"%(np.mean(list(ap_dic.values()))))
    
    def _interp_ap(self, precision, recall):

        if precision.size == 0 or recall.size == 0:
            return 0.

        iap = 0
        for r in np.arange(0.,1.1, 0.1):
            recall_mask = (recall >= r)
            p_max = precision[recall_mask]
            
            iap += np.max( p_max if p_max.size > 0 else [0] )

        return iap / 11


    def compute_ap(self, detections, num_gts):

        detections_sort_indx = np.argsort(-detections[:,1])
        detections = detections[detections_sort_indx]
        precision = []
        recall = []

        if num_gts == 0:
            return 0.

        for i in range(1, len(detections) + 1):
            precision.append( np.sum(detections[:i][:,2]) / i )
            recall.append( np.sum(detections[:i][:,2]) / num_gts )
        
        #print(f'Precision: {precision}')
        #print(f'Recall: {recall}')
        return self._interp_ap(np.array(precision), np.array(recall))
        
    def detect(self, image):
        
        # image_data = np.expand_dims(image, 0)
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image,
                self.input_image_shape: [self.input_shape[1], self.input_shape[0]],
                K.learning_phase(): 0
            })
        
        '''
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw '''
        
        return out_boxes, out_scores, out_classes


    def get_generator(self):
        val_split = 0.5
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val
        batch_size = 1

        data_get = self.data_generator_wrapper(lines[num_train:], batch_size, self.input_shape, self.anchors, self.num_classes)
        return data_get

    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        count = 0
        i = 0
        while count<n+2:
            image_data = []
            box_data = []
            for b in range(batch_size):
                #if i==0:
                #    np.random.shuffle(annotation_lines)
                image, box = self.get_random_data(annotation_lines[i], input_shape)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            bounding_boxes, class_labels = self.preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            count += 1
            yield image_data, bounding_boxes, class_labels

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n==0 or batch_size<=0: return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
    
    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        '''Preprocess true boxes to training input format
        Parameters
        ----------
        true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        Returns
        -------
        bounding_boxes: list of lists containing bounding boxes
        class_labels: list containing class labels
        '''
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
        true_boxes = np.array(true_boxes)
        true_boxes = true_boxes[~np.all(true_boxes==0, axis=1)]
        bounding_boxes = true_boxes[...,0:4]
        class_labels = true_boxes[...,4]
        '''
        num_layers = len(anchors)//3 # default setting
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0]>0

        for b in range(m):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh)==0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b,t, 4].astype('int32')
                        class_labels.append(c)
                        bounding_boxes.append(list(true_boxes[b,t,0:4]))
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

        print(f'Bounding boxes : {bounding_boxes}')
        print(f'Class labels: {class_labels}')
        '''
        return bounding_boxes, class_labels

    def get_classes(self, classes_path):
        '''loads the classes'''
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # Generate colors for drawing bounding boxes.
        '''
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        '''
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def _find_detection(self, q_box, boxes, global_index):

        if boxes.size == 0:
            #print('EMPTY BOXES')
            return -1

        ious = list(map(lambda x: self.compute_iou(q_box, x), boxes))

        max_iou_index = np.argmax( ious )

        if ious[max_iou_index] > self.iou:
            return global_index[max_iou_index]

        return -1

    def compute_iou(self, bb_1, bb_2):

        xa0, ya0, xa1, ya1 = bb_1
        xb0, yb0, xb1, yb1 = bb_2

        intersec = (min([xa1, xb1]) - max([xa0, xb0]))*(min([ya1, yb1]) - max([ya0, yb0]))

        union = (xa1 - xa0)*(ya1 - ya0) + (xb1 - xb0)*(yb1 - yb0) - intersec

        return intersec / union

    def get_random_data(self, annotation_line, input_shape, random=False, max_boxes=20, proc_img=True):
        '''random preprocessing for real-time data augmentation'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image_data=0
            if proc_img:
                image = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w,h), (128,128,128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image)/255.

            # correct boxes
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale #+ dx
            box[:, [1,3]] = box[:, [1,3]]*scale #+ dy
            box_data[:len(box)] = box

            return image_data, box_data


