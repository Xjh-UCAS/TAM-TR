from ultralytics import RTDETRWorld

if __name__ == '__main__':
    model = RTDETRWorld('TAM_TR.pt')
    
    # Define custom classes
    #model.set_classes(["car", "van"])

    # Execute prediction for specified categories on an image
    results = model.predict('/bigdata/XJH/yolov9/visdrone/VisDrone2019-DET-val/images/', 
                            save=True, 
                            conf=0.4, 
                            iou=0.6, 
                            name='TAMTR', 
                            project='runs/predict',
                            imgsz=640,
                            batch=4,)

