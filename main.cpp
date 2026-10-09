#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;

const int RESTORE_WIDTH = 640;
const int RESTORE_HEIGHT = 480;

void makeLetterbox(const Mat& source, Mat& output, float& scale, int& padLeft, int& padTop) {
    scale = min(INPUT_WIDTH / static_cast<float>(source.cols), INPUT_HEIGHT / static_cast<float>(source.rows));
    int newWidth = max(1, static_cast<int>(round(source.cols * scale)));
    int newHeight = max(1, static_cast<int>(round(source.rows * scale)));

    Mat resized;
    resize(source, resized, Size(newWidth, newHeight));

    int padWidth = INPUT_WIDTH - newWidth;
    int padHeight = INPUT_HEIGHT - newHeight;
    padLeft = padWidth / 2;
    padTop = padHeight / 2;
    int padRight = padWidth - padLeft;
    int padBottom = padHeight - padTop;

    copyMakeBorder(resized, output, padTop, padBottom, padLeft, padRight, BORDER_CONSTANT, Scalar(114, 114, 114));
}
void runYoloDetection(Net& yoloNet, const Mat& sourceFrame, Mat& displayFrame, const vector<string>& classNames, const vector<String>& outputNames) {
    Mat inputImage;
    float scale;
    int padLeft, padTop;

    makeLetterbox(sourceFrame, inputImage, scale, padLeft, padTop);

    Mat blob = blobFromImage(inputImage, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    yoloNet.setInput(blob);

    vector<Mat> outputs;
    yoloNet.forward(outputs, outputNames);

    Mat output = outputs[0];
    int dim1 = output.size[1];
    int dim2 = output.size[2];

    Mat raw(dim1, dim2, CV_32F, output.ptr<float>());
    Mat predictions;
    int classCount = classNames.size();

    if (dim1 == 4 + classCount) predictions = raw.t();
    else if (dim2 == 4 + classCount) predictions = raw;
    else return;

    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    for (int i = 0; i < predictions.rows; ++i) {
        const float* row = predictions.ptr<float>(i);
        float maxConfidence = 0.0f;
        int bestClass = -1;

        for (int j = 4; j < predictions.cols; ++j) {
            if (row[j] > maxConfidence) {
                maxConfidence = row[j];
                bestClass = j - 4;
            }
        }

        if (maxConfidence < CONF_THRESHOLD) continue;
        if (bestClass < 0 || bestClass >= classCount) continue;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        if (w <= 0 || h <= 0) continue;

        float x1 = (cx - w / 2 - padLeft) / scale;
        float y1 = (cy - h / 2 - padTop) / scale;
        float x2 = (cx + w / 2 - padLeft) / scale;
        float y2 = (cy + h / 2 - padTop) / scale;

        int left = max(0, min(static_cast<int>(floor(x1)), sourceFrame.cols - 1));
        int top = max(0, min(static_cast<int>(floor(y1)), sourceFrame.rows - 1));
        int right = max(0, min(static_cast<int>(ceil(x2)), sourceFrame.cols));
        int bottom = max(0, min(static_cast<int>(ceil(y2)), sourceFrame.rows));

        int boxWidth = right - left;
        int boxHeight = bottom - top;

        if (boxWidth <= 0 || boxHeight <= 0) continue;

        boxes.emplace_back(left, top, boxWidth, boxHeight);
        confidences.push_back(maxConfidence);
        classIds.push_back(bestClass);
    }

    vector<int> selectedIndices;
    for (int c = 0; c < classCount; ++c) {
        vector<Rect> classBoxes;
        vector<float> classConfidences;
        vector<int> originalIndices;

        for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
            if (classIds[i] == c) {
                classBoxes.push_back(boxes[i]);
                classConfidences.push_back(confidences[i]);
                originalIndices.push_back(i);
            }
        }

        if (classBoxes.empty()) continue;

        vector<int> localIndices;
        NMSBoxes(classBoxes, classConfidences, CONF_THRESHOLD, NMS_THRESHOLD, localIndices);

        for (int localIndex : localIndices) {
            selectedIndices.push_back(originalIndices[localIndex]);
        }
    }

    for (int idx : selectedIndices) {
        Rect box = boxes[idx];
        int id = classIds[idx];

        Scalar color = (id == 2) ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // tank is red

        rectangle(displayFrame, box, color, 2);
        int percent = static_cast<int>(round(confidences[idx] * 100.0f));
        string label = classNames[id] + " " + to_string(percent) + "%";
        int textY = (box.y >= 25) ? box.y - 8 : box.y + 22;

        putText(displayFrame, label, Point(box.x, textY), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 3, LINE_AA);
        putText(displayFrame, label, Point(box.x, textY), FONT_HERSHEY_SIMPLEX, 0.6, color, 2, LINE_AA);
    }
}

int main() {
    try {
        // YOLLAR
        string yoloModelPath = "...yolo8n_military.onnx";
        string restoreModelPath = ".../aodnet_480x640.onnx";
        string videoPath = ".../video.mp4";

        vector<string> classNames = {
            "camouflage_soldier", "weapon", "military_tank", "military_truck",
            "military_vehicle", "civilian", "soldier", "civilian_vehicle",
            "military_artillery", "trench", "military_aircraft", "military_warship"
        };

        cout << "Loading models..." << endl;
        Net restoreNet = readNetFromONNX(restoreModelPath);
        restoreNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        restoreNet.setPreferableTarget(DNN_TARGET_CPU);

        Net yoloNet = readNetFromONNX(yoloModelPath);
        yoloNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        yoloNet.setPreferableTarget(DNN_TARGET_CPU);

        VideoCapture cap(videoPath);
        if (!cap.isOpened()) return -1;

        vector<String> outputNames = yoloNet.getUnconnectedOutLayersNames();
        if (outputNames.empty()) return -1;

        Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            Mat originalDisplay = frame.clone();
            Mat restoreBlob = blobFromImage(frame, 1.0 / 255.0, Size(RESTORE_WIDTH, RESTORE_HEIGHT), Scalar(0, 0, 0), true, false);
            restoreNet.setInput(restoreBlob);
            Mat restoreOutput = restoreNet.forward();

            int h = restoreOutput.size[2];
            int w = restoreOutput.size[3];
            int spatialSize = h * w;
            Mat restoredMat(h, w, CV_32FC3);
            float* restoreData = (float*)restoreOutput.data;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    float r = max(0.0f, min(1.0f, restoreData[0 * spatialSize + idx]));
                    float g = max(0.0f, min(1.0f, restoreData[1 * spatialSize + idx]));
                    float b = max(0.0f, min(1.0f, restoreData[2 * spatialSize + idx]));
                    restoredMat.at<Vec3f>(y, x) = Vec3f(b, g, r);
                }
            }
            restoredMat.convertTo(restoredMat, CV_8UC3, 255.0);

            Mat finalRestoredFrame;
            resize(restoredMat, finalRestoredFrame, Size(frame.cols, frame.rows));
            Mat restoredDisplay = finalRestoredFrame.clone();

            runYoloDetection(yoloNet, frame, originalDisplay, classNames, outputNames);

            runYoloDetection(yoloNet, finalRestoredFrame, restoredDisplay, classNames, outputNames);

            putText(originalDisplay, "1. NORMAL", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
            putText(restoredDisplay, "2. CLEAN", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2, LINE_AA);

            Mat smallOriginal, smallRestored;

            resize(originalDisplay, smallOriginal, Size(originalDisplay.cols / 2, originalDisplay.rows / 2));
            resize(restoredDisplay, smallRestored, Size(restoredDisplay.cols / 2, restoredDisplay.rows / 2));

            imshow("NORMAL", smallOriginal);
            imshow("CLEAN", smallRestored);

            if (waitKey(1) == 27) break;
        }
        cap.release();
        destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV error: " << e.what() << endl;
    }
    return 0;
}