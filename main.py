from ultralytics import YOLO

# Load a model
#model = YOLO("data.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("D:/SaffronDetection/valid/images/287_jpg.rf.da1ec3021f6e60c2e697376454ad58a5.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format