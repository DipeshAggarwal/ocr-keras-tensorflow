import matplotlib
matplotlib.use("Agg")

from core.models import ResNet
from core.az_dataset.helpers import load_mnist_dataset
from core.az_dataset.helpers import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", default="a_z_handwritten_data.csv", help="Path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, default="handwriting.model", help="Path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output training history file")
args = vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-1
BS = 128

print("[INFO] Loading Datasets...")
az_data, az_labels = load_az_dataset(args["az"])
digits_data, digits_labels = load_mnist_dataset()

# Add 10 labels to AZ label to account for mnsit dataset
az_labels += 10

data = np.vstack([az_data, digits_data])
labels = np.hstack([az_labels, digits_labels])

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
data /= 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
counts = labels.sum(axis=0)

class_totals = labels.sum(axis=0)
class_weight = {}

for i in range(0, len(class_weight)):
    class_weight[i] = class_totals.max() / class_totals[i]
    
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
    )

print("[INFO] Compiling Model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(lb.classes_), (3, 3, 3), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training Network...")
H = model.fit(
    aug.flow(train_x, train_y, batch_size=BS),
    validation_data=(test_x, test_y),
    steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS,
    class_weight=class_weight,
    verbose=1)

label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_names = [l for l in label_names]

print("[INFO] Evaluating Network...")
predictions = model.predict(test_x, batch_size=BS)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

print("[INFO] Serialising Network...")
model.save(args["model"], save_format="h5")

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

images = []

for i in np.random.choice(np.arange(0, len(test_y), size=(49,))):
    probs = model.predict(test_x[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = label_names[prediction[0]]
    
    image = (test_x[i] * 255).astype("uint8")
    color= (0, 255, 0)
    
    if prediction[0] != np.argmax(test_y[i]):
        color = (0, 0, 255)
        
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    images.append(image)
    
montage = build_montages(images, (96, 96), (7, 7))[0]

cv2.imshow("OCR Results", montage)
cv2.waitKey(0)
