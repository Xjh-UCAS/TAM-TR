import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETRWorld

if __name__ == '__main__':
    model = RTDETRWorld('ultralytics/cfg/models/TAMTR/TAMTR.yaml')
    #model.load('') # loading pretrain weights
    model.train(data='dataset/visdrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=6,
                workers=8,
                device='0',
                #resume='/weights/last.pt', # last.pt path
                project='runs/train',
                name='TAMTR',
                )