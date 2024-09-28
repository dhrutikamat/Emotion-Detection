# Emotion-Detection using Deep Learning

# Project Overview
This project is an implementation of facial emotion detection using deep learning techniques. It leverages a pre-trained MobileNet model, along with OpenCV and Keras libraries, to accurately classify human emotions from real-time video frames captured through a webcam. The model can classify emotions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The primary objective of this project is to create a system that can recognize and analyze emotions, which can have practical applications in human-computer interaction, behavioral analysis, and customer service enhancements.

# Libraries Used
1. TensorFlow/Keras: For building and training the neural network.
2. OpenCV: For real-time video processing and face detection.
3. NumPy: For handling image data.
4. Matplotlib: For visualizing model performance.

# Project Structure and Functionality
● Model Training:
1. A MobileNet model pre-trained on the ImageNet dataset is used as the base.
2. The model is fine-tuned with a custom dataset of 28,709 images categorized into seven classes.
3. The dataset is augmented using data generators to increase variety.
4. The model is trained with callbacks such as Early Stopping and Model Checkpoints to ensure optimal performance and prevent overfitting.
5. The best model is saved as best_model.keras for deployment.

● Real-Time Emotion Detection:
1. The trained model is used to make predictions on video frames captured from the webcam.
2. OpenCV's Haar cascade classifier detects faces in each frame, and the region of interest (ROI) is extracted.
3. The ROI is resized and fed into the model to predict the emotion, which is then displayed on the screen.
   
● Visualization and Analysis:
1. Training and validation accuracy and loss are plotted to analyze model performance.
2. Sample images from the training data are displayed to visualize the input data.

# Result
With a well-trained model, the emotion detection system achieved promising results, effectively classifying emotions with high accuracy. The productivity impact of this system depends on its integration into real-world applications, providing meaningful insights and responses based on human emotions. The trained model achieves an accuracy of around 70% on the validation set.

