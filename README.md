## CVI techniques used to pre processing of input image
The input is a RGB image of crack. We use otsu thresholding on the image and dilate it to remove dark patches. ![image](https://user-images.githubusercontent.com/84932711/128039575-d85d6045-7bd2-4c7d-849c-67e021e56688.png)
Next we try edge detection techniques on the input image. We start with sobel edge detection. We find SobelX and SobelY which is basically the gradient of pixel intensity in x and y direction. ![image](https://user-images.githubusercontent.com/84932711/128039822-48976b8a-35e2-4eb5-9214-1967324e581d.png)
Using directional derivatives gives us specific cracks depending upon the direction of the gradient calculated. ![image](https://user-images.githubusercontent.com/84932711/128040706-cd21f201-39a8-4be1-9511-517df6abdc63.png)
To improve on this we find the second order derivative and find the laplacian of the image. After finding the laplacian we do basic morphology operations to remove noise and improve the result. ![image](https://user-images.githubusercontent.com/84932711/128040184-a6059c26-435c-453c-ae01-764515dfa429.png)
We looked at the image histograms trying to infer the drop in pixel values to find crack in images. We split the converted hsv image into h s and v channels and plot a histogram for all 3 channels. ![image](https://user-images.githubusercontent.com/84932711/128040437-6b9fc32c-5b71-4b24-bcb5-f801d2f3c9c0.png)
To get better insight on the image, we plot the 3d contours of hue saturation and value using matplotlib.mplot3d. 
![image](https://user-images.githubusercontent.com/84932711/128040621-acfec8a3-5823-4860-b9d3-7cb4d2fa2cb2.png)
![image](https://user-images.githubusercontent.com/84932711/128040645-1e6dda93-9761-4abb-9cf2-5451aa8ea11c.png)
![image](https://user-images.githubusercontent.com/84932711/128040654-6442cad8-ac90-4c92-95fe-09cf2a992098.png)
