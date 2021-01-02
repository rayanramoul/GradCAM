** GradCAM algorithm
*** Principle : 
GradCAM is an improved algorithm of CAM  (  that identifies regions discriminatively   )
In CAM, we weight these feature maps using weights taken out of the last fully-connected layer of the network. In Grad-CAM, we weight the feature maps using “alpha values” that are calculated based on gradients. Therefore, Grad-CAM does not require a particular architecture, because we can calculate gradients through any kind of neural network layer we want. The “Grad” in Grad-CAM stands for “gradient.”


*** Characteristics :
- Might be applied to many architectures of CNN : fully-connected ( VGG ) , structured output , multi modal inputs without needing to retrain it.
- Lack of interpretability of deep nets that might be compensated with GradCAM.
- Class Activation Mapping ( CAM ) : identification of discriminative regions used for the classification of a specific image class.
- GradCAM is a generalization of CAM applicable to + types of architectures
- It's good to capture in high resolution images fine grained detail.
- Localization approaches like CAM or GRAD-CAM are very discriminating: the explanation of "cat" only highlights the region linked to the cat and not the one linked to "dog".
- GradCAM might be useful to disriminate for the same prediction of 2 different models the difference in confidence of each one.
- It's useful for the visualization of classification process of popular architectures.
- CAM requires an architecture that applies global average pooling (GAP) to the final convolutional feature maps, followed by a single fully connected layer that produces the predictions
- Usage of neuron importance of Grad Cam and neuron names to extract a visual explanation of model's decisions.





** Steps of Grad-CAM :
*** Notations :
y^c : Output of the network depending of the class ( before softmax that transforms the output to a probability )
GradCAM is applied to a neural net which finished its training and with fixed weights, we take an image and feed it to the network and output the heatmap with GradCAM.

*** Step 1 : Calcul Gradient
Compute gradient of y^c relatively to the feature map activaiton A^k of last convolution layer. 
We get from a 2D input image a 3D gradient with the same shape as feature maps. 
K feature maps with each one a height of v and width u,  and final shape of [k, v, u], and so the gradient have the same shape.

*** Step 2 : Compute Alphas by mean of Gradients
Compute mean gradient on height and width axes, which gives a value of gradient per neuron then we multiply it times  the gradient computed in step 1 which gives us this equation :
[[./results/alpha_equation.png]]

Gradient Shape : [k, v, u], pooling on width and height and we finish with a shape of [k, 1, 1] or [k] alpha values.

*** Step 3 : Compute Final Grad-CAM Heatmap
We process this final equation : 
[[./results/l_gradcam.png]]

- The size of the final heatmap is [u, v], the same size as the final convolutional feature map, but we can upsample it to fit the size of the original image.

** Metrics
*** Average precision
To study the efficiency of object localization models we must use a metric insensitive to non-uniform distribution of the number of objects per classes, so using a simple accuracy metrics wouldn't work ( because if a class is really preponderant the model might learn to predict everything as being a part of this class and get good accuracy ) 
The precision as defined in the next equation mesaure the ratio of true positives or ratio of correctly detected objets compared to the overall number of objects classified.

Precision = True Positives /  ( True Positives + False Positives )




*** Recall
- The recall references the rate of false negatives or ratio of objects detected correctly compared to the overall number of objects classified.

Recall = True Positives / ( True positives + False Negatives )


- These two equations are linked to a threshold connected to the model which defines from which value (or confidence rate) the classifier will determine that an object belongs to a class.
- These two concepts are used in order to calculate a measure which has been introduced \ cite {pascal} in order to respond to the previously mentioned bias problems resulting in the measurement of average precision or AP , where  Recall_{i}  can take 11 values ​​ [0, 0.1, 0.2, ..., 1]  so we take the mean value of the precision on all the recall values.

AP = 1/11 * SUM_{Recall i} Precision(Recall_{i})


- The mAP will be the average of the average precision over all the object classes.
- However, the measures seen so far concern the problem of object classification, concerning that of object localization it is necessary to add a definition related to the surfaces of a bounding box and for that we use Localization and intersection on Union "Localization and Intersection over Union" it has the advantage of taking into account the models and types of predicted shapes (for example some models can locate objects in a rectangular area, andd other segmentation pixel by pixel) and this sums up to what extent the object to be predicted encroaches on the object predicted by the model as can be represented by the next figure.
[[./results/iou.png]]



** implementation
There are 3 scripts each one have its content commented : 
*** gradcam_detection.py :
Contains the classes and functions for GradCAM, and generate bounding boxes pickle files and score them.

*** gradcam_evaluate.py : 
Script that uses already saved data of gradcam with chosen thresholds and print their scores.

*** gradcam_random_visualization :
A script to visualize 10 random images with bounding boxes of ground truth + gradcam with a chosen threshold

- The is also a directory "results" containing : all_gt_dogs ( ground truth boxes ), and  all_detected_dogs for each threshold tested, those files help evaluate faster GradCAM ( with gradcam_evaluate.py )

** Results :
Example of output :
[[./results/gradcam_bbx.png]]

The results of the implementation depends of threshold selected for the heatmap extracted with GradCAM

- threshold = 0.02
    |        |       0.3 |       0.4 |       0.5 |
    |--------+-----------+-----------+-----------|
    | ap     | 46.853608 | 32.152346 | 17.796862 |
    | recall | 61.132075 | 50.566038 | 36.792453 |


- threshold = 0.01    
    |        |       0.3 |       0.4 |       0.5 |
    |--------+-----------+-----------+-----------|
    | ap     | 48.880971 | 34.132415 | 17.308526 |
    | recall | 62.075472 | 51.698113 | 36.226415 |

- threshold = 0.06   
    |        |       0.3 |       0.4 |       0.5 |
    |--------+-----------+-----------+-----------|
    | ap     | 47.716435 | 32.736706 | 18.843717 |
    | recall | 61.698113 | 50.566038 | 37.547170 |


** What could be better :
- By using a model trained on a classification task we can't really extract for each dog an entire bounding box ( especially when his entire body is visible ) because the model tends to find important features like : muzzle or fur and not the form of entire body.
- We could to improve this consider the localization as a markov decision process, take resnet50 as feature extractor, then a Deep-Q-Network that will take the output of resnet50 as input and learn to localize the objects in image, and then improve the precision of its results by applying GradCAM. 
- We could seek the best threshold for gradcam algorithm by looping on VOC train test and testing a range of values for it and find the one that maximizes the score.
- GradCAM looks also like an interesting algorithm for image ( pixels ) segmentation.
- Find a better radius for occlusion of a heatmap to obtain better wrapping of bounding box even with lower threshold values.