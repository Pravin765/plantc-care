---
license: mit
library_name: keras
pipeline_tag: image-classification
tags:
  - image-classification
  - plant-disease
  - agriculture
  - tflite
  - efficientnetv2
---

# Plantulus Doctor — plant disease classifier

EfficientNetV2-B0 (ImageNet pre-trained) fine-tuned to recognise **54 plant/disease classes across 19 plants** from a photo of a single leaf. Live demo and JSON API: [KumarjitC/indoor-plant-disease-gradio-app](https://huggingface.co/spaces/KumarjitC/indoor-plant-disease-gradio-app).

## Files

| File | Use |
|---|---|
| `plantulus_doctor_fp16.tflite` | Float16-quantized TFLite model (~12 MB) for servers (LiteRT) and mobile |
| `plantulus_doctor.keras` | Full Keras 3 model |
| `classes.json` | Class labels in model output order |
| `class_info.json` | Per class: label, plant, disease, `is_healthy`, display name |
| `preprocessing.json` | Required input preprocessing |

## Preprocessing

1. Apply EXIF orientation and convert to RGB.
2. Resize the shorter side to 256 px and centre-crop to 256×256.
3. Centre-crop to 224×224.
4. Feed float32 pixel values in **[0, 255]**. Do **not** divide by 255: the model normalizes internally.

Input shape `[1, 224, 224, 3]`; output: softmax probabilities over the 54 classes in `classes.json` order.

## Training data

- [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (colour images only)
- [New Plant Diseases](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (augmented PlantVillage)
- [Indoor Plant Disease Detection](https://www.kaggle.com/datasets/abdulahad0296/indoor-plant-disease-detection-dataset)

Images were deduplicated (MD5), and augmented copies were grouped by PlantVillage UUID and a rotation/flip-invariant perceptual hash so that copies of the same photo never appear in both train and test. Split ≈70/15/15, stratified by class.

Training: two stages (frozen backbone head training, then full fine-tuning with frozen BatchNorm and cosine learning-rate decay), random-crop/flip/rotation/zoom/colour augmentation, square-root class weights, label smoothing 0.1, mixed precision. Because of label smoothing, confident predictions typically score around 0.9 rather than 1.0.

## Evaluation (held-out test set, 16,697 images)

| Subset | Images | Accuracy | Macro F1 |
|---|---|---|---|
| **All** | 16,697 | **98.04%** | **0.984** |
| PlantVillage | 7,759 | 99.45% | 0.992 |
| New Plant Diseases | 5,919 | 99.27% | 0.936 |
| Indoor plants | 3,019 | 92.02% | 0.960 |

Top-3 accuracy: 99.75%.

## Limitations

- Most training images are single leaves on plain backgrounds; expect lower accuracy on cluttered field or household photos (the indoor subset is the best indicator).
- The model always predicts one of its 54 classes, even for unrelated images or unknown diseases; use the confidence score and top-3 output.
- Not a substitute for expert diagnosis.
