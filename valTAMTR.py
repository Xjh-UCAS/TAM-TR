import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETRWorld

if __name__ == '__main__':
    model = RTDETRWorld('TAM_TR.pt')

    model.val(data='dataset/visdrone.yaml',
              split='val',
              imgsz=640,
              batch=4,
              conf=0.4,
              iou=0.6,
              plots=True,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='TAMTR',
              )