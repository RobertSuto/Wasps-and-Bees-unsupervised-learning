# Image Clustering of Bees and Wasps

This project aims to cluster images of bees and wasps using unsupervised learning techniques. The dataset used contains 2127 pictures of wasps and 2469 pictures of bees, which were resized to 120 x 120 pixels. 

Two feature extraction methods were used: color spaces and bag of visual words (BOVW). For color spaces, the images were converted to different color spaces, and the mean of the n-channels was computed for all pixels in an image. The HSV color space was chosen as it gave slightly better results. For BOVW, SIFT descriptors were used to extract a collection of descriptors, which were then clustered using KMeans.

Three clustering algorithms were applied to the feature vectors: AgglomerativeClustering, KMeans, and MiniBatchKMeans. The performance of each algorithm was evaluated using both random chance and a supervised baseline. The results of the clustering algorithms were compared and presented in the documentation.

Overall, this project provides insights into the effectiveness of different feature extraction methods and clustering algorithms for image clustering tasks.
