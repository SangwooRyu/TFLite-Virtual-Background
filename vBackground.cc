#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <cmath>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

using namespace std;

int model_width;
int model_height;
int model_channels;

std::unique_ptr<tflite::Interpreter> interpreter;

void GetImageTFLite(float* out, cv::Mat &src);
void detect_from_video(cv::Mat &src);

struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

const RGB maskColor[2] = {{  0,   0,   0},  //background
                          {255, 255, 255}}; //person

void GetImageTFLite(float* out, cv::Mat &src)
{
    int i;
    float f;
    uint8_t *in;
    cv::Mat image;
    int Len;

    // copy image to input as input tensor
    cv::resize(src, image, cv::Size(model_width, model_height), cv::INTER_AREA);

    in=image.data;
    Len=image.rows*image.cols*image.channels();
    for(i=0;i<Len;i++){
        f     =in[i];
        out[i]=(f - 127.5f) / 127.5f;
    }
}

void detect_from_video(cv::Mat &src)
{
    int i,j,k,mi;
    float mx,v;
    float *data;
    RGB *rgb;
    static cv::Mat image;
    static cv::Mat frame(model_width, model_height, CV_8UC3);
    static cv::Mat blend(src.cols   , src.rows    , CV_8UC3);

    GetImageTFLite(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    interpreter->Invoke();      // run your model

    // get max object per pixel
    data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    rgb = (RGB *)frame.data;

    for(i = 0; i < model_height; i++){
        for(j = 0; j < model_width; j++){
            for(mi = -1, mx = 0.0, k = 0; k < 21; k++){
                v = data[21 * (i * model_width + j) + k];
                if(v > mx){ mi = k; mx = v; }
            }
            if(mi == 15)
                rgb[j + i * model_width] = maskColor[1];
            else
                rgb[j + i * model_width] = maskColor[0];
        }
    }

    //merge output into frame
    cv::Mat refined;
    cv::Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    
    cv::resize(frame, blend, cv::Size(src.cols,src.rows),cv::INTER_CUBIC);
    cv::morphologyEx(blend, refined, cv::MORPH_OPEN, mask);
    cv::morphologyEx(refined, refined, cv::MORPH_CLOSE, mask);
    blend = refined.clone();

    cv::GaussianBlur(blend, refined, cv::Size(0, 0), 3);
    cv::addWeighted(src, 0.5, refined, 0.5, 0.0, src);
}

int main(){
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("lite-model_deeplabv3_1_metadata_2.tflite");

    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();
    //interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(2);

    int In = interpreter->inputs()[0];
    model_height   = interpreter->tensor(In)->dims->data[1];
    model_width    = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];
    cout << "height   : "<< model_height << endl;
    cout << "width    : "<< model_width << endl;
    cout << "channels : "<< model_channels << endl;

    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cv::Mat frame;
    cv::VideoWriter outputVideo;
    cv::Size outSize = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	outputVideo.open("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, outSize, true);

    while(1){
        cap >> frame;

        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }

        detect_from_video(frame);

        outputVideo << frame;
    }

    /*cv::Mat src = cv::imread("test.jpg");
    detect_from_video(src);
    cv::imwrite("result.jpg", src);*/

    cap.release();
    outputVideo.release();

    return 0;
}