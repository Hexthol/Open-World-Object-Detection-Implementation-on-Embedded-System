import json
import cv2 as cv
import torch
import numpy as np
CLASS_NAME = ['measuring_spoons', 'small_spoon', 'bowl', 'measuring_cup_glass', 'timer', 'salt', 'hot_pad', 'measuring_cup_1/2', 'pan', 'oatmeal', 'big_spoon', 'distractors']




def nms(dets, scores,thresh):
    x1 = dets[:, 0] #xmin
    y1 = dets[:, 1] #ymin
    x2 = dets[:, 2]+x1 #xmax
    y2 = dets[:, 3]+y1 #ymax
    scores = scores #confidence
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort()[::-1] # boundingbox的置信度排序
    keep = [] # 用来保存最后留下来的boundingbox
    while order.size > 0:     
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]]) #交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]]) #交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]]) #交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]]) #交叉区域右下角的纵坐标
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep

def draw(imgfile,bboxes,savepath,name,objectnames):
    img = cv.imread(imgfile)
    
    for bbox,objectname in zip(bboxes,objectnames):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0]+bbox[2]
        y2 = bbox[1]+bbox[3]
        if(objectname == 'distractors'):
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
        else:
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=2)
        
        cv.putText(img, objectname, (int(x1), int(y1)), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
              thickness=1)
    cv.imwrite(savepath+'/'+name, img)  #save picture
    return img


with open('test.json') as f:
    result = json.load(f)

new_result = dict()
for i in range(len(result)):
    if(result[i]['score']<0.15):
        continue
    if str(result[i]['image_id']) not in new_result.keys(): 
        new_result[str(result[i]['image_id'])]=dict()
        new_result[str(result[i]['image_id'])]['bbox']=list()
        new_result[str(result[i]['image_id'])]['cls']=list()
        new_result[str(result[i]['image_id'])]['score']=list()
    new_result[str(result[i]['image_id'])]['bbox'].append(result[i]['bbox'])
    new_result[str(result[i]['image_id'])]['cls'].append(CLASS_NAME[result[i]['category_id']-1])
    new_result[str(result[i]['image_id'])]['score'].append(result[i]['score'])

for img in new_result.keys():
    img_path = 'image_6.png'
    save_path = './'
    name = 'test.jpg'
    bboxes = new_result[img]['bbox']
    objectnames = new_result[img]['cls']
    scores = new_result[img]['score']
    idx=nms(np.array(bboxes),np.array(scores),0.3)
    bboxes = torch.tensor(bboxes)
    new_obj = []
    for i in range(len(objectnames)):
        if i in idx:
            new_obj.append(objectnames[i])
    draw(imgfile=img_path,savepath=save_path,name=name,bboxes=bboxes[idx],objectnames=new_obj)
    
def final_draw(result,image_p,path,name):
    new_result = dict()
    for i in range(len(result)):
        if(result[i]['score']<0.3):
            continue
        if str(result[i]['image_id']) not in new_result.keys(): 
            new_result[str(result[i]['image_id'])]=dict()
            new_result[str(result[i]['image_id'])]['bbox']=list()
            new_result[str(result[i]['image_id'])]['cls']=list()
            new_result[str(result[i]['image_id'])]['score']=list()
        new_result[str(result[i]['image_id'])]['bbox'].append(result[i]['bbox'])
        new_result[str(result[i]['image_id'])]['cls'].append(CLASS_NAME[result[i]['category_id']-1])
        new_result[str(result[i]['image_id'])]['score'].append(result[i]['score'])
    res = cv.imread(image_p)
    for img in new_result.keys():
        img_path = image_p
        save_path = path
        name = name
        bboxes = new_result[img]['bbox']
        objectnames = new_result[img]['cls']
        scores = new_result[img]['score']
        idx=nms(np.array(bboxes),np.array(scores),0.3)
        bboxes = torch.tensor(bboxes)
        new_obj = []
        for i in range(len(objectnames)):
            if i in idx:
                new_obj.append(objectnames[i])
        res = draw(imgfile=img_path,savepath=save_path,name=name,bboxes=bboxes[idx],objectnames=new_obj)
    return res
