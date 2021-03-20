/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

struct stat st2;

//creating prototype function to run the main pipeline.
void feature_matching_pipeline_2D(string detector ,string descriptor,bool bVisDetect, bool bVisDescr , bool bVisMatch);
// creating prototype function to ensure new empty directories to insert outputdata.
void new_empty_directory (bool bVisDetect, bool bVisDescr , bool bVisMatch);

/* MAIN PROGRAM */

int main(int argc, const char *argv[])
{ 

  // creating vector to specify detector and descriptor
 
  vector<string> detectors_list = {"SHITOMASI","HARRIS","FAST","BRISK","ORB", "AKAZE", "SIFT"};
  vector<string> descriptors_list ={"BRISK","BRIEF","ORB","FREAK","AKAZE","SIFT"};
  // Creating boolean bVis variables to define if the images will be generated for each category (detector , descriptor and Matching)
  
  bool bVisDetect = true;
  bool bVisDescr = true;
  bool bVisMatch = true; 
  
  if(stat("../output",&st2) != 0)
  {
    mkdir( "../output" , 0777);
  }
   
  else
  {
    bVisDetect = false;
    bVisDescr = false;
    bVisMatch = false; 
    cout<<"##########################################################"<<endl;
    cout<<"### Please , it is necessary remove the output folder ####"<<endl;
    cout<<"##########################################################"<<endl;
  }
  
  new_empty_directory (bVisDetect, bVisDescr , bVisMatch);
  
  if (bVisDetect)
  {
    std::ofstream outfile ("../output/Detector_report.txt");
    for (string detect_it:detectors_list)
    {
      feature_matching_pipeline_2D(detect_it ,"", bVisDetect, false , false);
    }
  }

  if (bVisDescr)
  {
    std::ofstream outfile ("../output/Descriptor_report.txt");
    for (string detect_it :detectors_list)
    {
        
    string path_file = "../output/Detector_Descriptor/Det_"+ detect_it;
      
    const char* p_c_str = path_file.c_str();
      
    //checking and creating directory
    if(stat(p_c_str,&st2) != 0)
      mkdir( p_c_str , 0777);     

      
      for (string descrip_it :descriptors_list)
      {
        if ((detect_it == "SIFT")&& (descrip_it == "ORB"))
        {
         // do nothing
        }
        else if ((detect_it != "AKAZE")&&(descrip_it == "AKAZE"))
        {
         // do nothing
        }
        
        else
        {
          feature_matching_pipeline_2D(detect_it ,descrip_it, false, bVisDescr , false);
        }        
      }
    } 
  }
 
  if (bVisMatch)
  {             
    std::ofstream outfile ("../output/Matching_Descriptors_report.txt");
    for (string detect_it :detectors_list)
    {
     
    string path_file = "../output/Matching_Descriptors/Det_"+ detect_it;
      
    const char* p_c_str = path_file.c_str();
      
    //checking and creating directory
    if(stat(p_c_str,&st2) != 0)
      mkdir( p_c_str , 0777);     

      
      for (string descrip_it :descriptors_list)
      {
        if ((detect_it == "SIFT")&& (descrip_it == "ORB"))
        {
         // do nothing
        }
        else if ((detect_it != "AKAZE")&&(descrip_it == "AKAZE"))
        {
         // do nothing
        }
              
        else
        {
          feature_matching_pipeline_2D(detect_it ,descrip_it, false, false ,true);
        }        
      }
    } 
  }
 
  return 0; 
}


void feature_matching_pipeline_2D(string detectorChosen , string descriptorChosen, bool bVisDetect, bool bVisDescr , bool bVisMatch)
{
    // creating vector to specify detector and descriptor
    // detectors = SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    // descriptors = BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
   
  
    /* INIT VARIABLES AND DATA STRUCTURES */
  
    // data location
    string dataPath = "../"; // It is not necessary edit this line for the current strucutre folder.

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // miscelaneous 
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    //bool bVis = false;            // visualize results
    int img_index_num = 1;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
      

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if (dataBuffer.size() >  dataBufferSize)
        {
          dataBuffer.erase(dataBuffer.begin());
          dataBuffer.push_back(frame);    
        }
        else
        {
          dataBuffer.push_back(frame);
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = detectorChosen;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
      
        vector<string> otherfetures = {"FAST","BRISK","ORB", "AKAZE", "SIFT"};

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, img_index_num ,bVisDetect);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray,img_index_num ,bVisDetect);
        }
        else
        {
           if(std::find(otherfetures.begin(), otherfetures.end(), detectorType) != otherfetures.end())
           {
             detKeypointsModern(keypoints, imgGray, detectorType,img_index_num ,bVisDetect);
           }
           else           
           {
             cout<<"The Keypoint detector chosen it is not valid !! "<<endl;
           }
    
        }

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {

          keypoints.erase(remove_if(keypoints.begin(), keypoints.end(),
                                [&vehicleRect](const cv::KeyPoint& point) {
                                  return !vehicleRect.contains(point.pt);
                                }),
                      keypoints.end());
          cout << "Number of keypoints remaining after limiting to preceding "
              "vehicle: " << keypoints.size() << "\n";
          
        }
    
        // calculate distribution of neighborhood size
        double average_size = 0;
        for (auto kp : keypoints) {
            average_size += kp.size;
        }
        average_size /=  keypoints.size();

        std::cout << "Keypoints distribution of neighborhood size " << average_size << std::endl;
      
        if(bVisDetect)
        {
          fstream report;
          report.open("../output/Detector_report.txt", std::ios_base::app);
          report <<","<<keypoints.size()<<","<<average_size<<"\n";
          report.close();              
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        std::vector<string>descriptor_list = {"BRISK","BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
        //bVis = true;
        cv::Mat descriptors;
        string descriptorType = descriptorChosen; // BRIEF, ORB, FREAK, AKAZE, SIFT

        if(std::find(descriptor_list.begin(), descriptor_list.end(), descriptorType) != descriptor_list.end())
        {
          descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, detectorType,img_index_num ,bVisDescr);
        }
        else           
        {
          cout<<"The Descriptor algorithm chosen it is not valid !! "<<endl;
        }

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
      
            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp      
      
        
        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */
            
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN            
            string descriptorType_matching;
             
            //specific setting to solve opencv error for SIFT descriptor in function batchDistance
            if (descriptorType.compare("SIFT")== 0) 
            {
              descriptorType_matching = "DES_HOG"; 
            }
            else
            {
              descriptorType_matching = "DES_BINARY"; // DES_BINARY, DES_HOG
            }

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,(dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors, matches, descriptorType_matching, matcherType, selectorType , dataBuffer , detectorChosen , descriptorChosen , img_index_num, bVisMatch);

            //// EOF STUDENT ASSIGNMENT 

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image

        }
      
      img_index_num++;
      

    } // eof loop over all images

 
}

void new_empty_directory (bool bVisDetect, bool bVisDescr , bool bVisMatch)
{
 
   
  if (bVisDetect)
  {
    mkdir( "../output/Detector" , 0777);
  }
  if (bVisDescr)
  {
    mkdir("../output/Detector_Descriptor", 0777);
  }
  if ( bVisMatch)
  {
    mkdir("../output/Matching_Descriptors" , 0777);
  }            

}
