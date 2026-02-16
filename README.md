# Computer-Vision-Deep-Learning
This repository showcases advanced Computer Vision projects ranging from real-time object detection to face recognition, utilizing state-of-the-art architectures like YOLOv8 and InceptionResNet.

Deep Learning for Computer Vision 👁️
This repository showcases advanced Computer Vision projects ranging from real-time object detection to face recognition, utilizing state-of-the-art architectures like YOLOv8 and InceptionResNet.

🚀 Key Implementations
1. Object Detection (YOLOv8)
Custom Training: Fine-tuned YOLOv8 on a custom dataset to detect specific objects.

Augmentation Pipeline: Implemented a robust data augmentation strategy using torchvision.transforms.v2, including RandomResizedCrop, ColorJitter, and RandomRotation to prevent overfitting.

2. Face Recognition
Architecture: Utilized InceptionResNetV1 (pre-trained on VGGFace2) to generate 512-dimensional face embeddings.

Verification: Implemented a distance-based verification system (L2 Euclidean distance) to recognize specific individuals with a high confidence threshold.

3. Training Optimization
Transfer Learning: Leveraged pre-trained ImageNet weights to accelerate convergence on smaller datasets.

Callbacks: Implemented Early Stopping and Model Checkpointing to save the best model weights and prevent wasted compute resources.

🛠️ Tech Stack
Frameworks: PyTorch, Ultralytics (YOLO), TensorFlow/Keras.

Libraries: torchvision, facenet-pytorch, PIL.

Text-to-Video: Implemented CogVideoX using the diffusers library to generate short video clips from natural language prompts, demonstrating state-of-the-art Generative AI capabilities.
