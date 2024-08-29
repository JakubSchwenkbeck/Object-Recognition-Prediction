# Object Tracking in Java using OpenCV

This project is a Java application that utilizes OpenCV to capture video from a webcam, perform object tracking and movement prediction, and display the video feed in real-time. The program tests key components like OpenCV library loading, camera initialization, frame capture, and image display using `HighGui`.

## Features

- **OpenCV Integration**: Uses OpenCV for real-time video processing.
- **HighGui Display**: Renders video frames with object tracking on the screen.
- **Object Tracking**: Drawing bounding boxes around detected objects with classification.

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
    - Execute the `TestObjectTracking` class.
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

