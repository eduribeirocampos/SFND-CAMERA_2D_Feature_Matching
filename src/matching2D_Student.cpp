#include <numeric>
#include "matching2D.hpp"

using namespace std;  

struct stat st;

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, int img_index,bool bVisDetect)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


  
    // visualize results
    if (bVisDetect)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        //imshow(windowName, visImage);

        ostringstream saving_name;
        saving_name <<img_index<<"_SHITOMASI"<<"_n=" << keypoints.size() << "_t=" << 1000 * t / 1.0 << "_ms";
        string file_name = saving_name.str();    

        string path_file = "../output/Detector/SHITOMASI/";

        // creating directory with detector type name
        const char* p_c_str = path_file.c_str();
      
        //checking and creating directory
        if(stat(p_c_str,&st) != 0)
          mkdir( p_c_str , 0777);
  
        imwrite(path_file +"img"+ file_name+".jpg", visImage);       
        //cv::waitKey(0);
      
        fstream report;
        report.open("../output/Detector_report.txt", std::ios_base::app);
        report <<"SHITOMASI,"<<img_index <<","<<keypoints.size() <<","<<1000 * t / 1.0 ;
        report.close();        

    }
}

// Detect keypoints using the harris corners
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, int img_index ,bool bVisDetect) {
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::Mat dst;

    double t = (double)cv::getTickCount();
    cv::cornerHarris( img, dst, blockSize, apertureSize, k );
    cv::Mat dst_norm;
    cv::Mat dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    int threshold = 100;
    for (int i=0; i < img.rows; ++i) {
        for (int j=0; j < img.cols; ++j) {
            float response = dst_norm.at<float>(i,j);
            if ((int) response > threshold) {
                cv::KeyPoint testKp;
                testKp.pt = cv::Point2f(j, i);
                testKp.size = 2*apertureSize;
                testKp.response = response;

                bool overlap = false;
                for (auto it=keypoints.begin(); it != keypoints.end(); it++) {
                  if (cv::KeyPoint::overlap(testKp, *it) > 0) {
                    overlap = true;
                    if (testKp.response > it->response) {
                      *it = testKp;
                      break;
                    }
                  }
                }
                
                if (!overlap) {
                    keypoints.push_back(testKp);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris Corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  

    // visualize results
    if (bVisDetect)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corners Detector Results";
        cv::namedWindow(windowName, 6);
        //imshow(windowName, visImage);
      
        ostringstream saving_name;
        saving_name <<img_index<<"_HARRIS"<<"_n=" << keypoints.size() << "_t=" << 1000 * t / 1.0 << "_ms";

        string file_name = saving_name.str();    
    
  
    
        string path_file = "../output/Detector/HARRIS/";

        // creating directory with detector type name
        const char* p_c_str = path_file.c_str();
      
        //checking and creating directory
        if(stat(p_c_str,&st) != 0)
          mkdir( p_c_str , 0777);
  
        imwrite(path_file +"img"+ file_name+".jpg", visImage);          
      
        fstream report;
        report.open("../output/Detector_report.txt", std::ios_base::app);
        report <<"HARRIS,"<<img_index <<","<<keypoints.size() <<","<<1000 * t / 1.0;
        report.close();      
            
        //cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, int img_index ,bool bVisDetect)
{
  cv::Ptr<cv::FeatureDetector> detector;
  if (detectorType == "FAST")
  {
    int threshold = 30;    // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;      // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
  }
  else if (detectorType == "BRISK")
  {
    int threshold = 30;        //   AGAST detection threshold score
    int octaves = 3;           // detection octaves
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
    detector = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if (detectorType == "ORB")
  {
    int   nfeatures = 500;     // The maximum number of features to retain.
    float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
    int   nlevels = 8;         // The number of pyramid levels.
    int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
    int   firstLevel = 0;      // The level of pyramid to put source image to.
    int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
    auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
    int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
    int   fastThreshold = 20;  // The FAST threshold.
    detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (detectorType == "AKAZE")
  {
    // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
    //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
    int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
    float threshold = 0.001f;         //   Detector response threshold to accept point.
    int   nOctaves = 4;               // Maximum octave evolution of the image.
    int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
    auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
    //                   DIFF_WEICKERT or DIFF_CHARBONNIER.
    detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                 threshold, nOctaves, nOctaveLayers, diffusivity);
  }
  else if (detectorType == "SIFT")
  {
    int nfeatures = 0; // The number of best features to retain.
    int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
    // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
    double contrastThreshold = 0.04;
    double edgeThreshold = 10; // The threshold used to filter out edge-like features.
    double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave \#0.

    detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
  }
  else
  {
    throw std::invalid_argument("unknown detector type: " + detectorType);
  }

  auto t = static_cast<double>(cv::getTickCount());
  detector->detect(img, keypoints);
  t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
  cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    
  if (bVisDetect)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Keypoint Detector Results";
    cv::namedWindow(windowName, 6);
    //imshow(windowName, visImage);
    
    ostringstream saving_name;
    saving_name <<img_index<<"_"<<detectorType<<"_n=" << keypoints.size() << "_t=" << 1000 * t / 1.0 << "_ms";
    
    string file_name = saving_name.str();    
    
    ostringstream folder;
    folder <<detectorType;
    string folder_name = folder.str();       
    
    string path_file = "../output/Detector/" + folder_name + "/";

    // creating directory with detector type name
    const char* p_c_str = path_file.c_str();
      
    //checking and creating directory
    if(stat(p_c_str,&st) != 0)
      mkdir( p_c_str , 0777);
  
    imwrite(path_file + "img"+ file_name+".jpg", visImage);    

    fstream report;
    report.open("../output/Detector_report.txt", std::ios_base::app);
    report <<detectorType<<","<<img_index <<","<<keypoints.size() <<","<<1000 * t / 1.0;
    report.close();      
    
    
    //cv::waitKey(0);
  }
} 

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType , string detectorType , int img_index , bool bVisDescr)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
      int bytes = 32;               // legth of the descriptor in bytes
      bool use_orientation = false; // sample patterns using key points orientation

      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
      
      
    }  
    else if (descriptorType.compare("ORB") == 0)
    {

      extractor = cv::ORB::create();

    }  
    else if (descriptorType.compare("AKAZE") == 0)
    {

      extractor = cv::AKAZE::create();

    }   
    else if (descriptorType.compare("SIFT") == 0)
    {


    extractor = cv::xfeatures2d::SIFT::create();
      
    }   
    else if (descriptorType.compare("FREAK") == 0)
    {

      extractor = cv::xfeatures2d::FREAK::create();

    }    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
  
    cout << descriptorType << " with n= " << keypoints.size() << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

  
 
    if (bVisDescr)
  { 
  
      
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Keypoint Detector Results";
    cv::namedWindow(windowName, 6);
    //imshow(windowName, visImage);
    
    ostringstream saving_name;
    saving_name <<img_index<<"_Detec_"<<detectorType<<"_Desc_"<<descriptorType<<"_n=" << keypoints.size() << "_t=" << 1000 * t / 1.0 << "_ms";
    string file_name = saving_name.str(); 
      
 
    ostringstream folder;
    folder <<"Det_"<<detectorType<<"_Desc_"<<descriptorType;
    string folder_name = folder.str();   
      
   
      
    string path_file = "../output/Detector_Descriptor/Det_"+detectorType+"/" +folder_name+"/";
      
    const char* p_c_str = path_file.c_str();
      
    //checking and creating directory
    if(stat(p_c_str,&st) != 0)
      mkdir( p_c_str , 0777);
     

    imwrite(path_file +"/"+ file_name+".jpg", visImage);
    
    fstream report;
    report.open("../output/Descriptor_report.txt", std::ios_base::app);
    report <<detectorType<<","<<descriptorType <<","<<img_index <<","<<keypoints.size() <<","<<1000 * t / 1.0 << '\n';
    report.close(); 

      
    //cv::waitKey(0);
  }
  
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, std::vector<DataFrame> dataBuffer, std::string detector, std::string descriptor,int img_index,bool bVisMatch )
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t;
    int KNN_keypoints_removed = 0;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        //specific setting to solve opencv error for SIFT descriptor in function batchDistance
        if (descriptorType.compare("DES_HOG") ==0)
        {
           normType = cv::NORM_L2;
        }
      
        else
        {
          normType = cv::NORM_HAMMING;
        }
      
        matcher = cv::BFMatcher::create(normType, crossCheck);

    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      
      if (descSource.type() != CV_32F)
      { // OpenCV bug workaround :
        // convert binary descriptors to floating point due to a bug in current OpenCV implementation
        descSource.convertTo(descSource, CV_32F);
      }
      

      if (descRef.type() != CV_32F)
      {
        descRef.convertTo(descRef, CV_32F);
      }

      matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);        

      }

  
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        t = (double)cv::getTickCount();

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
      
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
    cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;     

    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
      int k = 2; // number of neighbours
      vector<vector<cv::DMatch>> knn_matches;
      
      t = (double)cv::getTickCount();

      matcher->knnMatch(descSource, descRef, knn_matches, k); // finds the k best matches

      t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
      cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;      
      // filter matches using descriptor distance ratio test
      double minDescDistRatio = 0.8;
      for (auto& knn_match : knn_matches)
      {
        if (knn_match[0].distance < minDescDistRatio * knn_match[1].distance)
        {
          matches.push_back(knn_match[0]);
        }
      }
      KNN_keypoints_removed = knn_matches.size() - matches.size();
      cout << "(KNN) # keypoints removed = " << KNN_keypoints_removed << endl;   
    }
    if (bVisMatch)
      {
          cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
          cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                          (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,                            matches, matchImg,
                           cv::Scalar::all(-1), cv::Scalar::all(-1),
                           vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
              
              
          string windowName = "Matching keypoints between two camera images";
          cv::namedWindow(windowName, 7);
          //cv::imshow(windowName, matchImg);
          //cout << "Press key to continue to next image" << endl;
         // cv::waitKey(0); // wait for key to be press   
          
          ostringstream saving_name;
          saving_name <<detector<<"_"<<descriptor<<"_"<<img_index<<"_n=" << matches.size() << "_t=" << 1000 * t / 1.0 << "_ms";
          string file_name = saving_name.str();    

          ostringstream folder;
          folder <<"Det_"<<detector<<"_Desc_"<<descriptor;
          string folder_name = folder.str();   
 
          string path_file = "../output/Matching_Descriptors/Det_"+detector+"/" +folder_name+"/";        
        
          // creating directory with detector type name
          const char* p_c_str = path_file.c_str();
      
          //checking and creating directory
          if(stat(p_c_str,&st) != 0)
            mkdir( p_c_str , 0777);
  
          imwrite(path_file +"/"+ file_name+".jpg", matchImg);          
      
          fstream report;
          report.open("../output/Matching_Descriptors_report.txt", std::ios_base::app);
          report <<detector<<","<<descriptor<<","<<img_index<<"," << matches.size() << "," << 1000 * t / 1.0 <<","<<KNN_keypoints_removed<<'\n';
          report.close();      
    }

}