[//]: # (Image References)
 
[image1]: ./210319_output/Detector/BRISK/img9_BRISK_n=2639_t=46.9851_ms.jpg
[image2]: ./210319_output/Detector_Descriptor/Det_FAST/Det_FAST_Desc_ORB/10_Detec_FAST_Desc_ORB_n=143_t=1.05644_ms.jpg
[image3]: ./210319_output/Matching_Descriptors/Det_BRISK/Det_BRISK_Desc_SIFT/BRISK_SIFT_3_n=193_t=3.39979_ms.jpg
[image4]: ./images/Descriptor_evaluation.png
[image5]: ./images/Det_BRISK_Descriptor_evaluation.png
[image6]: ./images/Det_ORB_Descriptor_evaluation.png
[image7]: ./images/Det_SIFT_Descriptors_evaluation.png
[image8]: ./images/Det_FAST_Descriptors_evaluation.png
[image9]: ./images/Descriptor_BRIEF_Detectors_evaluation.png

[image10]: ./images/Detector_FAST.gif
[image11]: ./images/Detector_FAST_Descriptor_BRIEF.gif
[image12]: ./images/Mathing_Detector_FAST_Descriptor_BRIEF.gif
[image13]: ./images/Detector_SIFT.gif



# SFND 2D Feature Tracking



## Objective and Overview

In this "Feature tracking" project, you will implement a few detectors, descriptors, and matching algorithms. The aim of having you implement various detectors and descriptors combinations is to present you with a wide range of options available in the OpenCV.

The project consists of four parts:

### 1. The Data Buffer: 
Start with loading the images, setting up the data structure, and put everything into the data buffer.

### 2. Keypoint Detection:
Integrate several keypoint detectors, such as HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT, and compare them to each other based on the number of key points and speed.

![alt text |width=450px | align="middle"][image10]

### 3. Descriptor Extraction & Matching: 
Extract the descriptors and match them using the brute-force and FLANN approach.

![alt text |width=450px | align="middle"][image11]
![alt text |width=450px | align="middle"][image12]

### 4. Performance Evaluation:
Compare and evaluate which combination of algorithms perform the best concerning performance measurement parameters.

In real-world practice also, you would need to implement a wide variety of detectors and descriptors (which can be done quickly with the OpenCV) and apply them to the images relevant to the project you are working on (e.g., pedestrian detection, aerial images).

## Goals

To achieve the goals of this project , all the PROJECT SPECIFICATION point covered on the [Rubric](https://review.udacity.com/#!/rubrics/2549/view) must be done.

## Main Files 

### Implementation Code:
- [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) main file.
- [matching2D_Student.cpp](src/matching2D_Student.cpp) support functions file;

### Running the code:

```
* Go to ../build
* execute ./2D_feature_tracking
```
The code will generate automatically the folder `output` containing all the information necessary to evaluate all the tasks required to achieve the Project goals.

On the freeze [210319_output](210319_output/) folder you will find all the images results and the .txt files with the report used to write the README file.

## Part 1 - Data Buffer.

### MP.1 - Data Buffer Optimization.
* Implemented on the file [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) from lines 190 to 198.

## Part 2 - Keypoints.

### MP.2 - Keypoint Detection.
* Implemented over the functions `detKeypointsShiTomasi`,`detKeypointsHarris` and `detKeypointsModern` on the file [matching2D_Student.cpp](src/matching2D_Student.cpp) from lines 9 to 257. And call in the main code [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) from line 215 to 232.

**`Detector:`** All The detectors list was implemented for all [Dataset images](images/KITTI/2011_09_26/image_00/data/) available.The next image is a example of the result.

file:[img9_BRISK_n=2639_t=46.9851_ms.jpg](210319_output/Detector/BRISK/img9_BRISK_n=2639_t=46.9851_ms.jpg)

![alt text |width=450px | align="middle"][image1]

### MP.3 - Keypoint Removal.
* Implemented on the file [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) from lines 242 to 255.

The quantity of the remaining points are available on the report file [Descriptor_report.txt](210319_output/Descriptor_report.txt). e.g line 237:`ORB,FREAK,7,66,40.6813`, For ORB Detector feature , the image 7 have 66 keypoints to process on image 7.

## Part 3 - Descriptors.

### MP.4 - Keypoint Descriptors.

* Implemented over the function `descKeypoints`on the file [matching2D_Student.cpp](src/matching2D_Student.cpp) from lines 260 to 357. And call in the main code [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) from line 285 to 295.

**`Descriptors:`** All The Descriptors list was implemented for all [Dataset images](images/KITTI/2011_09_26/image_00/data/) available.The next image is a example of the result (FAST+ORB).

file:[10_Detec_FAST_Desc_ORB_n=143_t=1.05644_ms.jpg](210319_output/Detector_Descriptor/Det_FAST/Det_FAST_Desc_ORB/10_Detec_FAST_Desc_ORB_n=143_t=1.05644_ms.jpg)

![alt text |width=450px | align="middle"][image2]

### MP.5 - Descriptor Matching

* Implemented over the function `matchDescriptors`on the file [matching2D_Student.cpp](src/matching2D_Student.cpp) from lines 359 to 480 And call in the main code [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) from line 307 to 329.

**`Matching:`** All The matching Descriptors list was implemented for all [Dataset images](images/KITTI/2011_09_26/image_00/data/) available.The next image is a example of the result (BRISK+SIFT) mathing points between image 3 and the previous image 2.

file:[BRISK_SIFT_3_n=193_t=3.39979_ms.jpg](210319_output/Matching_Descriptors/Det_BRISK/Det_BRISK_Desc_SIFT/BRISK_SIFT_3_n=193_t=3.39979_ms.jpg)

![alt text |width=450px | align="middle"][image3]

### MP.6 - Descriptor Distance Ratio

For this task it was edited the line 311 from the code [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) to set the parameter `selectorType = "SEL_KNN"`and in the [matching2D_Student.cpp](src/matching2D_Student.cpp)it was changed the line 420 to setup k nearest neighbors variable. it was made for k=1 k=2 and k=3.
The results could be find on the folder renamed as [knn_output_k1](knn_output_k1/), [knn_output_k2](knn_output_k2/) and [knn_output_k3](knn_output_k3/). The threshold is set to 0.8 for all condition to have the same base for comparison.


In order to reduce the generated data, on the file [MidTermProject_Camera_Student.cpp](src/MidTermProject_Camera_Student.cpp) in the lines 39 and 40   the booleans variables bVisDetect and  bVisDescr was set to false;

The table below shows the mean of removed bad keypoints for each descriptor. For k=1, we do not have Matching points between the scenes considerable accepted, for k=2 we have a good number of mathing points e a lower number of the removed bad keypoints. when k=3, the number of mathing poinhts couls be considered the same of k=2. It was made tests por k=4 and the number of mathing keypoints remaing the same.
So the `best Distance Ratio` to be considered is k=2.

<html>
<table>
<tr><th><center> k=1 <center></th><th><center> k=2 <center></th><th><center> k=3 <center><th></tr>
<tr><td>
 

| Detector for k=1   |  Match  |  removed_keypoints  |
|:------------------:|:-------:|:-------------------:|
| AKAZE              |    0    |         165         |
| BRISK              |    2    |         272         |
| FAST               |    1    |         148         |
| HARRIS             |    0    |         23          |
| ORB                |    0    |         101         |
| SHITOMASI          |    1    |         117         |
| SIFT               |    1    |         137         |
    
</td><td>

| Detector for k=2   |  Match  |  removed_keypoints  |
|:------------------:|:-------:|:-------------------:|
| AKAZE              |   136   |         29          |
| BRISK              |   176   |         97          |
| FAST               |   110   |         38          |
| HARRIS             |   17    |          6          |
| ORB                |   72    |         30          |
| SHITOMASI          |   95    |         22          |
| SIFT               |   74    |         63          |
    
</td><td>

| Detector for k=3   |  Match  |  removed_keypoints  |
|:------------------:|:-------:|:-------------------:|
| AKAZE              |   136   |         29          |
| BRISK              |   176   |         97          |
| FAST               |   110   |         38          |
| HARRIS             |   17    |          6          |
| ORB                |   72    |         30          |
| SHITOMASI          |   95    |         22          |
| SIFT               |   74    |         63          |

</td></tr> </table> </html> 

For the complete evaluation, please check the `notebook file`: [knn_Evaluation](knn_Evaluation.ipynb)

## Part 4 - Performance.
    
### MP.7 - Performance Evaluation 1.
    
**task** - count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

The next scatter plot shows the distribuion of the detected keypoints on the preceding vehicle and time elapsed.

![alt text |width=450px | align="middle"][image4]    
    
The next table shows the meaning of total keypoints detected in the scene on the column `keypoints` and the remaining points on the preceding vehicle on the column `pc`, the meaning of the distribution of the neighborhood size is the column `nhs`.
<br />      
<table>
<tr><th><center> Meaning values over the 10 images. <center><th></tr>
<tr><td>

 
| Detector   |  keypoints  |  time   |  pc  |  nhs  |
|:----------:|:-----------:|:-------:|:----:|:-----:|
| AKAZE      |    1342     | 129.023 | 167  |   7   |
| BRISK      |    2711     | 48.5482 | 276  |  21   |
| FAST       |    1787     | 1.05449 | 149  |   7   |
| HARRIS     |     173     | 19.6915 |  24  |   6   |
| ORB        |     500     | 8.49676 | 116  |  56   |
| SHITOMASI  |    1342     | 21.8832 | 117  |   4   |
| SIFT       |    1386     | 179.739 | 138  |   5   |

</td></tr> 
</table>     
<br />  
 
The BRISK feature detect a higher number of the Keypoints in a short elapsed time. the FAST feature shows capable to be fast and detect a good number of keypoints comparing with the others features.
    
For the complete evaluation, please check the `notebook file`: [Detector_Evaluation](Detector_Evaluation.ipynb)

### MP.8 - Performance Evaluation 2.
    
**task** - count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, use the BF approach with the descriptor distance ratio set to 0.8.
    
To make easy the undestand about the visualization, the next plots shows de evaluation for each matching descriptord over the 10 images and a unique Detector. The upper Plot is the overall evaluation with all matching descriptors, the lowest plot with the Y scale adjusted shows with more precision the quantity  and how fast  the matching descriptors process are.
    
The next plot shows the comparison for the Detector `BRISK`.
![alt text |width=450px | align="middle"][image5]    

    
The next plot shows the comparison for the Detector `ORB`.
![alt text |width=450px | align="middle"][image6]  

    
the next tables show the meaning of the 10 images for each combinatios (Detector , Descriptor) in the mathing descriptor process.
    
<html>
<table>
<tr><th><center> AKAZE Detector <center></th><th><center> BRISK Detector <center></th><th><center> FAST Detector <center><th></tr>
<tr><td>

|                    |  Matched  |  time   |
|:-------------------|:---------:|:-------:|
| ('AKAZE', 'AKAZE') |    139    | 236.069 |
| ('AKAZE', 'BRIEF') |    140    | 130.101 |
| ('AKAZE', 'BRISK') |    135    | 131.945 |
| ('AKAZE', 'FREAK') |    131    | 182.407 |
| ('AKAZE', 'ORB')   |    131    | 132.112 |
| ('AKAZE', 'SIFT')  |    141    | 174.808 |
    
</td><td>

|                    |  Matched  |  time   |
|:-------------------|:---------:|:-------:|
| ('BRISK', 'BRIEF') |    189    | 51.2361 |
| ('BRISK', 'BRISK') |    174    | 53.7704 |
| ('BRISK', 'FREAK') |    169    | 101.149 |
| ('BRISK', 'ORB')   |    168    | 54.9815 |
| ('BRISK', 'SIFT')  |    182    | 127.206 |
    
</td><td>

|                   |  Matched  |  time   |
|:------------------|:---------:|:-------:|
| ('FAST', 'BRIEF') |    122    | 2.6383  |
| ('FAST', 'BRISK') |    99     | 3.87659 |
| ('FAST', 'FREAK') |    97     | 50.5926 |
| ('FAST', 'ORB')   |    119    | 2.94983 |
| ('FAST', 'SIFT')  |    116    | 36.4681 |

</td></tr> </table> </html>
    
<html>
<table>
<tr><th><center> HARRIS Detector <center></th><th><center> ORB Detector <center></th><th><center> SHITOMASI Detector <center><th></tr>
<tr><td>
 

|                     |  Matched  |  time   |
|:--------------------|:---------:|:-------:|
| ('HARRIS', 'BRIEF') |    19     | 20.2939 |
| ('HARRIS', 'BRISK') |    15     | 20.5106 |
| ('HARRIS', 'FREAK') |    15     | 68.2798 |
| ('HARRIS', 'ORB')   |    17     | 20.7489 |
| ('HARRIS', 'SIFT')  |    18     | 37.7643 |
    
</td><td>


|                  |  Matched  |  time   |
|:-----------------|:---------:|:-------:|
| ('ORB', 'BRIEF') |    60     | 9.18033 |
| ('ORB', 'BRISK') |    83     | 10.2213 |
| ('ORB', 'FREAK') |    46     | 57.4886 |
| ('ORB', 'ORB')   |    84     | 14.2491 |
| ('ORB', 'SIFT')  |    84     | 98.0274 |
    
</td><td>

|                        |  Matched  |  time   |
|:-----------------------|:---------:|:-------:|
| ('SHITOMASI', 'BRIEF') |    104    | 20.5507 |
| ('SHITOMASI', 'BRISK') |    85     | 21.4426 |
| ('SHITOMASI', 'FREAK') |    85     | 69.4351 |
| ('SHITOMASI', 'ORB')   |    100    | 20.8391 |
| ('SHITOMASI', 'SIFT')  |    103    | 40.344  |

</td></tr> </table> </html>  
    
<br />      
<table>
<tr><th><center> SIFT Detector. <center><th></tr>
<tr><td>
    
|                   |  Matched  |  time   |
|:------------------|:---------:|:-------:|
| ('SIFT', 'BRIEF') |    78     | 177.52  |
| ('SIFT', 'BRISK') |    65     | 178.539 |
| ('SIFT', 'FREAK') |    65     | 228.678 |
| ('SIFT', 'SIFT')  |    88     | 310.648 |
    
</td></tr> 
</table>     
<br /> 
    
For the complete evaluation, please check the `notebook file`: [Descriptor_Matching_Evaluation](Descriptor_Matching_Evaluation.ipynb)    
    
    
### MP.9 - Performance Evaluation 1.
    
**task** - Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

the next tables show the meaning of the 10 images for each combinatios (Detector , Descriptor).
    
<html>
<table>
<tr><th><center> AKAZE Detector <center></th><th><center> BRISK Detector <center></th><th><center> FAST Detector <center><th></tr>
<tr><td>
    
|                    |  keypoints  |  time   |
|:------------------:|:-----------:|:-------:|
| ('AKAZE', 'AKAZE') |     167     | 235.876 |
| ('AKAZE', 'BRIEF') |     167     | 130.27  |
| ('AKAZE', 'BRISK') |     167     | 131.996 |
| ('AKAZE', 'FREAK') |     167     | 182.217 |
| ('AKAZE', 'ORB')   |     167     | 132.302 |
| ('AKAZE', 'SIFT')  |     167     | 173.683 |

    
</td><td>
    
|                    |  keypoints  |  time   |
|:------------------:|:-----------:|:-------:|
| ('BRISK', 'BRIEF') |     276     | 49.7287 |
| ('BRISK', 'BRISK') |     276     | 52.1583 |
| ('BRISK', 'FREAK') |     256     | 99.9105 |
| ('BRISK', 'ORB')   |     276     | 53.515  |
| ('BRISK', 'SIFT')  |     276     | 124.392 
    
</td><td>
   
|                   |  keypoints  |  time   |
|:-----------------:|:-----------:|:-------:|
| ('FAST', 'BRIEF') |     149     | 1.94531 |
| ('FAST', 'BRISK') |     149     | 3.12418 |
| ('FAST', 'FREAK') |     149     | 49.8411 |
| ('FAST', 'ORB')   |     149     | 2.21175 |
| ('FAST', 'SIFT')  |     149     | 35.0501 |
</td></tr> </table> </html>
    
<html>
<table>
<tr><th><center> HARRIS Detector <center></th><th><center> ORB Detector <center></th><th><center> SHITOMASI Detector <center><th></tr>
<tr><td>
    
|                     |  keypoints  |  time   |
|:--------------------|:-----------:|:-------:|
| ('HARRIS', 'BRIEF') |     24      | 20.1869 |
| ('HARRIS', 'BRISK') |     24      | 20.3743 |
| ('HARRIS', 'FREAK') |     24      | 68.0079 |
| ('HARRIS', 'ORB')   |     24      | 20.6234 |
| ('HARRIS', 'SIFT')  |     24      | 37.4531 |
    
</td><td>


|                  |  keypoints  |  time   |
|:-----------------|:-----------:|:-------:|
| ('ORB', 'BRIEF') |     116     | 9.32313 |
| ('ORB', 'BRISK') |     107     | 10.2791 |
| ('ORB', 'FREAK') |     62      | 58.1484 |
| ('ORB', 'ORB')   |     116     | 14.1574 |
| ('ORB', 'SIFT')  |     116     | 95.5988 |
    
</td><td>

|                        |  keypoints  |  time   |
|:-----------------------|:-----------:|:-------:|
| ('SHITOMASI', 'BRIEF') |     117     | 22.7427 |
| ('SHITOMASI', 'BRISK') |     117     | 23.5921 |
| ('SHITOMASI', 'FREAK') |     117     | 71.4313 |
| ('SHITOMASI', 'ORB')   |     117     | 22.9785 |
| ('SHITOMASI', 'SIFT')  |     117     | 42.0174 |

</td></tr> </table> </html>  
    
<br />      
<table>
<tr><th><center> SIFT Detector. <center><th></tr>
<tr><td>
    
|                   |  keypoints  |  time   |
|:------------------|:-----------:|:-------:|
| ('SIFT', 'BRIEF') |     138     | 180.658 |
| ('SIFT', 'BRISK') |     138     | 181.639 |
| ('SIFT', 'FREAK') |     137     | 231.786 |
| ('SIFT', 'SIFT')  |     138     | 313.413 |
    
</td></tr> 
</table>     
<br /> 
    
To make easy the undestand about the visualization, the next plots shows de evaluation for each descriptor over the 10 images and a unique Detector. The upper Plot is the overall evaluation with all descriptors, the lowest plot with the Y scale adjusted shows with more precision the quantity and how fast is the detector + descriptors process are.
    
The next plot shows the comparison for the Detector `SIFT`.
![alt text |width=450px | align="middle"][image7]    

    
The next plot shows the comparison for the Detector `FAST`.
![alt text |width=450px | align="middle"][image8]  

    

<br />     
For the complete evaluation, please check the `notebook file`: [Descriptor_Evaluation](Descriptor_Evaluation.ipynb)  
     
    
    
After check all the results, it is possible identify that the BRIEF feature had processed the descriptor with the lowest elapsed time. So lets compare all the Detectors together with only the BRIEF descriptor feature setup.
    
![alt text |width=450px | align="middle"][image9]      


## Conclusion.
    
The BRIEF descriptor feature shows the best choice for the most combined Detectors also regard matching points between the scenes(MP.8). On the last plot we can see that ,the combined BRISK+BRIEF has the highest number of descriptor points detected on de preceding vehicle and the processing time is something around 50 mili seconds maximum. Depends of the application this combination could be chosen, but we have a group of combinations that are capable to process the descriptor points under the 25 millisecond. If the application request a high frequency to read the data and this quantity of the descriptors keypoints is enough sufficient, the combination of FAST+BRIEF could be chosen. The results of AKAZE+BRIEF and SIFT+BRIEF must be avoided because the elapsed time necessary is to large comparing with others and HARRIS + BRIEF also must be avoided, it has a good performance regard time , but the number of descriptor keypoints processed is very small.    
    
    
    
    
    
    
    
    
    
    
    
    

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)