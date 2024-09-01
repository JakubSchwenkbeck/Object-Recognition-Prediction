# Object Tracking in Java using OpenCV (Work in progress)

This project is a Java application that utilizes **OpenCV** to capture video from a webcam, perform object tracking and movement prediction, and display the video feed in real-time. It uses key components such as OpenCV library loading, camera initialization, frame capture, and image display using **HighGui**. Additionally, the application integrates a custom Convolutional Neural Network (**CNN**) written from **scratch** in Java to classify and track objects in the video feed.

![image](https://github.com/user-attachments/assets/660b0931-ad11-417b-86f9-d04adeb78551)![image](https://github.com/user-attachments/assets/ae83c22d-dc77-4101-b0d2-f8c1e5d07aab)



## Features

- **OpenCV Integration**: Uses OpenCV for real-time video processing.
- **HighGui Display**: Renders video frames with object tracking on the screen.
- **Object Tracking**: Drawing bounding boxes around detected objects with classification.
- **Custom CNN Model**: Utilizes a self-written Convolutional Neural Network (CNN) implemented from scratch in Java for object classification.

### OpenCV for Java
**OpenCV** (Open Source Computer Vision Library) provides tools and libraries for real-time computer vision. This project leverages OpenCV for:

- **Video Capture**: Capturing live video from the webcam using the `VideoCapture` class.
- **Image Processing**: Performing real-time image processing tasks such as resizing, color space conversion, and feature detection.
- **Object Detection**: Applying custom object detection algorithms or pre-trained models to identify objects within video frames.
- **Image Display**: Using the `HighGui` class to create windows and display images and video streams in real-time.
- **Mouse and Keyboard Handling**: Responding to user input for interaction with the video feed.

### Own CNN:
The custom Convolutional Neural Network (CNN) implemented in Java is designed to classify and track objects from the video feed. Below are the key details and components of the CNN:
1. **Convolutional Layers**:
   - **Purpose**: Extract features from input images through convolution operations.
   - **Details**: The network includes multiple convolutional layers with varying filter sizes, strides, and padding to capture different levels of abstraction in the image.
   - **Example**: A layer with 32 filters of size 3x3, followed by a ReLU activation function.

2. **Pooling Layers**:
   - **Max Pooling**: Reduces spatial dimensions while retaining the most significant features. For example, 2x2 max pooling layers are used to downsample feature maps.
   - **Average Pooling**: Used in conjunction with max pooling to average feature values over local regions, enhancing the robustness of feature extraction.

3. **Fully Connected Layers**:
   - **Purpose**: Flatten the output from convolutional and pooling layers and make final predictions based on extracted features.
   - **Details**: Includes one or more fully connected layers that output the final class scores and bounding box coordinates.

- Training/Testing Data : [PASCAL VOC Data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf)
    
## Requirements

- **Java Development Kit (JDK)**: Ensure you have JDK 8 or higher installed.
- **OpenCV Library**: Download and set up OpenCV for Java.

## Setup Instructions

1. **Download and Install OpenCV**:
    - Download OpenCV from the [official website](https://opencv.org/).
    - Follow the instructions to set up OpenCV for Java, including adding the `opencv-<version>.jar` file to your project and configuring the native library path.

2. **Clone the Repository**:
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

3. **Open the Project**:
    - Open the project in your preferred IDE (e.g., IntelliJ IDEA or Eclipse).
    - Add the OpenCV JAR file to your project's build path.
    - Set the VM options to include the native library path:
      ```plaintext
      -Djava.library.path=<path_to_opencv_native_libs>
      ```

4. **Run the Application**:
    - Execute the `Main` class.
    - The application will test each component and display the video feed with object tracking.

## Usage

- The application captures video from your default webcam and displays it in a window.
- A sample bounding box is drawn on the video feed for demonstration purposes.
- Press the 'q' key to exit the application.

## Troubleshooting

- **Camera Not Found**: Ensure your webcam is connected and try using a different camera index.
- **Library Load Error**: Verify that the OpenCV native libraries are correctly configured in your project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

