## CVI techniques used for pre processing of input image
The input is a RGB image of crack. We use otsu thresholding on the image and dilate it to remove dark patches.                                                                                                                                                 
![image](https://user-images.githubusercontent.com/84932711/128039575-d85d6045-7bd2-4c7d-849c-67e021e56688.png) 



Next we try edge detection techniques on the input image. We start with sobel edge detection. We find SobelX and SobelY which is basically the gradient of pixel intensity in x and y direction.

![image](https://user-images.githubusercontent.com/84932711/128039822-48976b8a-35e2-4eb5-9214-1967324e581d.png) 


Using directional derivatives gives us specific cracks depending upon the direction of the gradient calculated. 
![image](https://user-images.githubusercontent.com/84932711/128040706-cd21f201-39a8-4be1-9511-517df6abdc63.png)



To improve on this we find the second order derivative and find the laplacian of the image. After finding the laplacian we do basic morphology operations to remove noise and improve the result. 


![image](https://user-images.githubusercontent.com/84932711/128040184-a6059c26-435c-453c-ae01-764515dfa429.png) 


We looked at the image histograms trying to infer the drop in pixel values to find crack in images. We split the converted hsv image into h s and v channels and plot a histogram for all 3 channels. 


![image](https://user-images.githubusercontent.com/84932711/128040437-6b9fc32c-5b71-4b24-bcb5-f801d2f3c9c0.png) 



To get better insight on the image, we plot the 3d contours of hue saturation and value using matplotlib.mplot3d. 
![image](https://user-images.githubusercontent.com/84932711/128040621-acfec8a3-5823-4860-b9d3-7cb4d2fa2cb2.png)
![image](https://user-images.githubusercontent.com/84932711/128040645-1e6dda93-9761-4abb-9cf2-5451aa8ea11c.png)
![image](https://user-images.githubusercontent.com/84932711/128040654-6442cad8-ac90-4c92-95fe-09cf2a992098.png)


We convert the original image to HSV. Now we create a mask such that only the crack is filtered out from the hsv image. We try all channels but get the most promising results from value channel as the crack is nothing but just a drop in intensity of the pixels. 
![image](https://user-images.githubusercontent.com/84932711/128045965-9edf096a-4d71-4105-8017-57388a6ad32a.png)
![image](https://user-images.githubusercontent.com/84932711/128045978-d71e2d43-b532-4249-883e-9d933c4818b3.png)


We select few different types of crack images from the dataset and apply the mask on all of them to verify the mask performance.
![image](https://user-images.githubusercontent.com/84932711/128046171-aa716d82-7b08-48d2-a097-71c173b678d8.png)
![image](https://user-images.githubusercontent.com/84932711/128046239-af021ce2-a669-4988-ac98-c1bc9e9a97d8.png)
![image](https://user-images.githubusercontent.com/84932711/128046302-4be7572e-3f3c-426d-9c3b-6702447a6707.png)
![image](https://user-images.githubusercontent.com/84932711/128046311-e2ac5b5b-4d2a-4249-b796-4bacdd725c3f.png)
![image](https://user-images.githubusercontent.com/84932711/128046317-e7ae37ef-cc2a-4ef1-994b-4cce53fdbaac.png)
![image](https://user-images.githubusercontent.com/84932711/128046329-24da59ab-ac52-42a5-aeb8-fc0d9db1e8d7.png)
![image](https://user-images.githubusercontent.com/84932711/128046344-a978ffb3-2072-49b3-bf29-947f20f7c1e9.png)
![image](https://user-images.githubusercontent.com/84932711/128046368-5d136639-73f9-44ca-8873-d16139241058.png)
![image](https://user-images.githubusercontent.com/84932711/128046381-ae3449f8-66c4-457b-bb9b-07bfb22768f8.png)
![image](https://user-images.githubusercontent.com/84932711/128046455-7b1f8e7a-0672-4963-a506-fedb367ad8b5.png)



To improve the quality of input image we perform contrast stretching on the rgb crack.


![image](https://user-images.githubusercontent.com/84932711/128046672-7e7d44d1-4d63-4ab8-974a-8f268fe73336.png)
![image](https://user-images.githubusercontent.com/84932711/128047825-584658ed-633c-40a9-90c8-c9604d6dab7a.png)
![image](https://user-images.githubusercontent.com/84932711/128047665-85c5ee77-1512-4148-857d-7bb2b4799b39.png)
![image](https://user-images.githubusercontent.com/84932711/128047671-7cf2c91d-bf59-484c-bedb-735708578de7.png)





We do contrast stretching and then apply filter on the contrasted crack. Here are the results:
![image](https://user-images.githubusercontent.com/84932711/128046873-6432cd94-dcd6-4389-ac6e-bbe6f60ba4fb.png)
![image](https://user-images.githubusercontent.com/84932711/128046888-e9161c47-7584-4e30-9224-3a5d2dd2d537.png)



To detect noise in the image that might hinder with the mask, we use blob detection to find crack like looking noise objects. First we use the direct blobdetector without tweaking the parameters:


![image](https://user-images.githubusercontent.com/84932711/128047140-04d5defd-b6a2-4cdf-9586-7523f6f65382.png)
![image](https://user-images.githubusercontent.com/84932711/128047154-2dcc5669-d31c-4251-a915-7e95faf8ec68.png)
![image](https://user-images.githubusercontent.com/84932711/128047169-acea76bf-bb63-49d3-aa0c-596c867e9f5d.png)
![image](https://user-images.githubusercontent.com/84932711/128047123-f86b4c44-5dcd-4793-9fe4-afc9df2534d2.png)


We tweak the parameters to get best results, like changing area to minimum, setting convexity of blob to 0.1, etc. 
![image](https://user-images.githubusercontent.com/84932711/128047379-21ce22e5-f65b-40a9-8942-33f17b1259d9.png)
![image](https://user-images.githubusercontent.com/84932711/128047435-ea7374e8-4a99-4886-8d82-1751c6a913a1.png)
![image](https://user-images.githubusercontent.com/84932711/128047447-908c0d81-b5f8-4b04-86a7-05bd0f4dd4f3.png)
![image](https://user-images.githubusercontent.com/84932711/128047455-6c4b4e15-cd79-4c6a-bc39-29a7bd05d1cd.png)

