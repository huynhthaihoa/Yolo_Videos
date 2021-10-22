#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "INIReader.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


//constexpr float CONFIDENCE_THRESHOLD = 0;
//constexpr float NMS_THRESHOLD = 0.4;
//constexpr int NUM_CLASSES = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

//typedef struct BoundingBox
//{
//    int xmin;
//    int xmax;
//    int ymin;
//    int ymax;
//};

void Yolo(cv::Mat frame, int NUM_CLASSES, int CONFIDENCE_THRESHOLD, int NMS_THRESHOLD, std::vector<cv::Mat> detections, std::vector<std::vector<int>>& indices, std::vector<std::vector<cv::Rect>>& boxes, std::vector<std::vector<float>>& scores)
{
    //detect
#if 0
    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        //std::cout << num_boxes << std::endl;
        for (int i = 0; i < num_boxes; ++i)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence > CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
#else   
    for (size_t i = 0; i < detections.size(); ++i)
    {
        float* data = (float*)detections[i].data;
        for (size_t j = 0; j < detections[i].rows; ++j, data += detections[i].cols)
        {
            cv::Mat score = detections[i].row(j).colRange(5, detections[i].cols);
            cv::Point classIdPoint;
            double confidence;

            // Get the value and location of the maximum score
            minMaxLoc(score, 0, &confidence, 0, &classIdPoint);
            
            if (confidence > CONFIDENCE_THRESHOLD)
            {
                auto x = (float)(data[0] * frame.cols);
                auto y = (float)(data[1] * frame.rows);
                auto width = (float)(data[2] * frame.cols);
                auto height = (float)(data[1] * frame.rows);
                cv::Rect rect(x - width / 2, y - height / 2, width, height);
                int c = classIdPoint.x;
                boxes[c].push_back(rect);
                scores[c].push_back(confidence);
            }

        }
    }
#endif

    //non-maximum suppress
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        //std::cout << "Size before NMS: " << boxes[c].size() << std::endl;
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        //std::cout << "Size after NMS: " << boxes[c].size() << std::endl;
    }
}

void updateStandardSize(int imgWidth, int imgHeight, int size, cv::Rect &origin)
{
    //cv::Rect crop(origin);
    if (size < origin.width)
        size = origin.width;
    if (size < origin.height)
        size = origin.height;

    int rWidthLeft = (size - origin.width) / 3;
    int rHeightTop = (size - origin.height) / 3;
    origin.x -= rWidthLeft;
    if (origin.x < 0)
        origin.x = 0;
    origin.width = size;
    if (origin.x + size >= imgWidth)
        origin.x = imgWidth - size;
    origin.y -= rHeightTop;
    if (origin.y < 0)
        origin.y = 0;
    origin.height = size;
    if (origin.y + size >= imgHeight)
        origin.y = imgHeight - size;
    //return crop;
}

int getFrameNum(const cv::String &filename)
{
    cv::VideoCapture cap(filename);
    int count = 0;
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) 
        {
            break;
        }
        ++count;
    }
    return count;
}

int countDigits(int nFrames)
{
    int count = 0;
    while (nFrames > 0)
    {
        ++count;
        nFrames /= 10;
    }
    return count;
}

std::string getSuffix(const int& nDigits, const int& index)
{
    int testNum = 10;// *nDigits;
    int count = 1;
    //while (testNum > index)
    //{
    //    ++count;
    //    testNum /= 10;
    //}
    while (testNum <= index)
    {
        ++count;
        testNum *= 10;
    }

    count = nDigits - count;

    std::string suffix = "";
    for (int i = 0; i < count; ++i)
    {
        suffix += "0";
    }
    suffix += std::to_string(index);
    return suffix;
}

int main()
{
    INIReader ini("config.ini");

    /*PARAMS section*/
    float CONFIDENCE_THRESHOLD = ini.GetReal("PARAMS", "confidence_threshold", 0.5);
    float NMS_THRESHOLD = ini.GetReal("PARAMS", "non_maximum_suppresion_threshold", 0.4);

    /*MODEL section*/
    cv::String classPath = ini.GetString("MODEL", "class", "classes.txt"); //"model.names";
    cv::String cfgPath = ini.GetString("MODEL", "config", ""); //"model.cfg";
    cv::String weightPath = ini.GetString("MODEL", "weight", ""); //"model.weights";
    
    /*INOUT section*/
    cv::String inPath = ini.GetString("INOUT", "input", ""); //folder contains input videos
    cv::String outPath = ini.GetString("INOUT", "output", ""); //folder contains extracting frames & their corresponding annotation files
    cv::String cropPath = ini.GetString("INOUT", "crop", ""); //folder contains cropped images of detection result
    cv::String logPath = ini.GetString("INOUT", "log", "logs.txt"); //log file
    bool isStandard = ini.GetBoolean("INOUT", "standard", true); //if isStandard is true, each cropped image will have a standard size, elsewise its depends on the corresponding bounding box
    int cropSize = ini.GetInteger("INOUT", "size", 608); //standard size of each cropped image (for standard mode)
    long ENTRY = ini.GetInteger("PARAMS", "entry", 0); //the first index for the first extract image
    long FREQUENCY = ini.GetInteger("PARAMS", "frequency", 1); //the frequency to save the extract image
    bool isSplit = ini.GetBoolean("INOUT", "split", true); //if isSplit is true, for each video, one subfolder will be created inside outPath to contain images extracted from this video; elsewise every image will be saved in outPath
    
    std::cout << CONFIDENCE_THRESHOLD << " " << NMS_THRESHOLD << "\n";

    int NUM_CLASSES = 0;
    std::vector<std::string> class_names;

    std::ifstream class_file(classPath);
    if (!class_file)
    {
        std::cerr << "failed to open classes.txt\n";
        return 0;
    }

    std::string line;
    while (std::getline(class_file, line))
    {
        class_names.push_back(line);
        ++NUM_CLASSES;
    }

    auto net = cv::dnn::readNetFromDarknet(cfgPath, weightPath);

#if 1
    /*GPU mode*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
    /*CPU mode*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif

    auto output_names = net.getUnconnectedOutLayersNames();

    std::ofstream log_file(logPath);

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;

    /*Video source*/
    std::vector<cv::String> videonames;
    std::vector<cv::String> extensions = { "mp4", "h264"};//, "png"
    std::vector<cv::String> currentExtStr;
    for (int i = 0; i < extensions.size(); ++i)
    {
        cv::glob(inPath + "/*." + extensions[i], currentExtStr);
        videonames.insert(videonames.end(), currentExtStr.begin(), currentExtStr.end());
    }

    size_t nVideos = videonames.size();
    log_file << "The number of videos is " << nVideos << std::endl;
    std::cout << "The number of videos is " << nVideos << std::endl;

    //cv::VideoCapture source(inPath);
    
    /*Create output folder*/
    if (cv::utils::fs::exists(outPath) == false)
        cv::utils::fs::createDirectory(outPath);
    if (cv::utils::fs::exists(cropPath) == false)
        cv::utils::fs::createDirectory(cropPath);
    cropPath += "/";
    
    cv::String subdirname; //directory of cropped images

    cv::String basename; //image file name

    cv::String basename2;

    //basename = outPath + "/" + basename.substr(0, basename.find_last_of(".")) + "_";

    cv::String outputPath;
    cv::String imgFilePath; //image file name
    cv::String annFilePath; //annotation file name

    double dRows; // 1 per image number of rows
    double dCols; // 1 per image number of columns
    double x;
    double y;

    long k; //for iterating frames in each video

    int obIdx; //for iterating object in each frame

    //bool isDetected; //check if container code is detected
    
    for (size_t i = 0; i < nVideos; ++i)
    {
        cv::VideoCapture source(videonames[i]);

        int numFrames = getFrameNum(videonames[i]);//source.get(cv::CAP_PROP_FRAME_COUNT);

        int numDigits = countDigits(numFrames);

        log_file << "Analyzing video " << videonames[i] << std::endl;

        std::cout << "Analyzing video " << videonames[i] << std::endl;

        basename = videonames[i].substr(videonames[i].find_last_of("\\") + 1); //video basename

        //if(prefix != "")
        //    basename = prefix + "_" + basename.substr(0, basename.find_last_of("."));
        //else
       basename = basename.substr(0, basename.find_last_of("."));

       basename2 = basename;

       if (isSplit == true)
       {
           subdirname = outPath + "/" + basename;

            if (cv::utils::fs::exists(subdirname) == false)
                cv::utils::fs::createDirectory(subdirname);

            basename = basename + "/" + basename;
       }

        k = 0;

        while (true)
        {
            source >> frame;

            if (k % FREQUENCY != 0)
                continue;

            cv::Size s = frame.size();

            if (frame.empty())
            {
                std::cout << "No more frame in video " << videonames[i] << "...\n";
                break;
            }

            dRows = 1. / frame.rows;
            dCols = 1. / frame.cols;

            outputPath = outPath + "/" + basename + "_" + getSuffix(numDigits, k + ENTRY);

            //std::string getSuffix(const int& nDigits, const int& index)
            annFilePath = outputPath + ".txt";
            imgFilePath = outputPath + ".png";
            //annFilePath = outPath + "/" + basename + "-" + std::to_string(k) + ".txt";
            //imgFilePath = outPath + "/" + basename + "-" + std::to_string(k) + ".png";
            //annFilePath = subdirname + "/" + basename + "_" + std::to_string(k) + ".txt";
            //imgFilePath = subdirname + "/" + basename + "_" + std::to_string(k) + ".png";



            log_file << "image: " << imgFilePath << std::endl;
            std::cout << "image: " << imgFilePath << std::endl;
            std::cout << "annotation: " << annFilePath << std::endl;

            std::ofstream annoFile(annFilePath, std::ofstream::out);
            if (frame.empty())
            {
                cv::waitKey();
                source.release();
                break;
            }

            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);

            auto dnn_start = std::chrono::steady_clock::now();
            try
            {
                net.forward(detections, output_names);
            }
            catch (std::exception& ex)
            {
                log_file << "We got a problem!: " << ex.what() << std::endl;
                break;
            }
            auto dnn_end = std::chrono::steady_clock::now();

            std::vector<std::vector<int>> indices(NUM_CLASSES);
            std::vector<std::vector<cv::Rect>> boxes(NUM_CLASSES); //bounding boxes
            std::vector<std::vector<float>> scores(NUM_CLASSES); //confidence scores

            Yolo(frame, NUM_CLASSES, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, detections, indices, boxes, scores);

            obIdx = 0;

            //isDetected = false;

            //label classes
            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                cv::String fold = cropPath + class_names[c];
                if (cv::utils::fs::exists(fold) == false)
                    cv::utils::fs::createDirectory(fold);
                
                if (isSplit == true)
                {
                    if (cv::utils::fs::exists(fold + "/" + basename2) == false)
                        cv::utils::fs::createDirectory(fold + "/" + basename2);
                }

                fold += "/";

                for (size_t j = 0; j < indices[c].size(); ++j)
                {
                    const auto color = colors[c % NUM_COLORS];

                    auto idx = indices[c][j];
                    auto& rect = boxes[c][idx];

                    /*Crop image*/
                    try
                    {
                        if (rect.x < 0)
                        {
                            rect.width += rect.x;
                            rect.x = 0;
                        }
                        if (rect.x + rect.width >= s.width)
                            rect.width = s.width - rect.x;
                        if (rect.y < 0)
                        {
                            rect.height += rect.y;
                            rect.y = 0;
                        }
                        
                        if (rect.y + rect.height >= s.height)
                            rect.height = s.height - rect.y;

                        cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
                        if (c != 0)
                            updateStandardSize(frame.cols, frame.rows, cropSize, roi);
                        //cv::Rect cropRect = getCroppedRegion(frame.cols, frame.rows, cropSize, roi);
                        
                        cv::Range rows(roi.x, roi.x + roi.width);
                        cv::Range cols(roi.y, roi.y + roi.height);
                        
                        cv::Mat cropRoi = frame(cols, rows);
                        
                        cv::String name = fold + basename + "_" + getSuffix(numDigits, k + ENTRY) + "_" + std::to_string(obIdx) + ".png";
                        
                        /*Only crop container code img*/
                        //if (c == 0 || c == 1)
                        //{
                            //if (!isDetected)
                            //    isDetected = true;
                            //cv::imwrite(name, cropRoi);
                        //}
                        cv::imwrite(name, cropRoi);
                    }
                    catch (const std::exception &ex)
                    {
                        log_file << "Error while cropping image: " << ex.what() << std::endl;
                    }
                    
                    /*Write to annotation file*/
                    try
                    {
                        //if (c != 0)
                        //    continue;
                        x = rect.x + rect.width / 2.0;// -1;
                        y = rect.y + rect.height / 2.0;// -1;

                        annoFile << std::to_string(int(c)) << " " << x * dCols
                            << " " << y * dRows
                            << " " << rect.width * dCols
                            << " " << rect.height * dRows << "\n";
                    }
                    catch (const std::exception& ex)
                    {
                        log_file << "Error while editing annotation file: " << ex.what() << std::endl;
                    }

                    ++obIdx;
                    log_file << class_names[c] << ":" << rect.x << " " << rect.y << std::endl;

                }
            }

            annoFile.close();
            ++k;
            //if(isDetected) //only save image which has detected objects
            cv::imwrite(imgFilePath, frame);
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "Finished!";
    return 0;
}