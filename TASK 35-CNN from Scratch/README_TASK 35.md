# **Task 35: Convolutional Neural Networks (CNNs)**


# Object Detection with CNN from Scratch

In this project, I created a Convolutional Neural Network (CNN) from scratch to perform various image processing tasks. The goal was to manually apply custom convolution filters to images, achieving tasks like edge detection, image sharpening, and embossing.

## Features

- **Edge Detection**: I implemented horizontal and vertical edge detection filters to highlight changes in pixel intensity.
- **Image Sharpening**: A custom filter was designed to sharpen the image by emphasizing pixel differences.
- **Emboss Effect**: I applied an embossing filter to give the image a 3D shadow and highlight effect.
- **Batch Processing**: The images were resized, normalized, and batch processed to make them suitable for CNN operations.



![oi](https://github.com/user-attachments/assets/4e1295ae-2398-449f-b2a5-c533f05c9dfe)


![ri](https://github.com/user-attachments/assets/64bdca8a-fad0-4afb-987a-a798769eb3b5)






## Prerequisites

Before running the project, make sure you have the following Python libraries installed:
- **NumPy**: For handling numerical computations.
- **OpenCV**: For image loading and manipulation.
- **Matplotlib**: For visualizing the results.

You can install these libraries using `pip`:
```bash
pip install numpy opencv-python matplotlib
```

## How It Works

1. **Image Loading and Preprocessing**:
   - I loaded the input image and resized it to a fixed 512x512 resolution for consistent processing.
   - The pixel values of the image were normalized to the range [0, 1] to ensure effective processing by the filters.

2. **Convolution Filters**:
   - I defined some custom convolution filters:
     - **Edge Detection Filters**: To detect horizontal and vertical edges.
     - **Sharpness Filter**: To enhance the clarity of the image.
     - **Emboss Filter**: To create a 3D-like effect by emphasizing edges and depth.
  
3. **Manual Convolution**:
   - I implemented a custom `conv2d` function that applies the filters to the image manually, without relying on pre-built CNN layers.
   - This function performs the actual convolution operation, where the filters are applied to the image pixel by pixel.

4. **Visualization**:
   - Using **Matplotlib**, I visualized the results of each filter applied to the image.
   - The processed images (edge detection, sharpening, embossing) were displayed alongside the original image for comparison.


![se](https://github.com/user-attachments/assets/05842fed-994f-4da9-bf2b-fe6b8a0f2580)



## Results

The project resulted in several processed images, including:
- Images with detected horizontal and vertical edges.
- A sharpened version of the image.
- An embossed image with a 3D effect.

I was able to observe how each filter impacts the image and how they can be used for various image processing tasks.


