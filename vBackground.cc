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
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#include "transpose_conv_bias.h"

using namespace std;

int model_width;
int model_height;
int model_channels;

std::unique_ptr<tflite::Interpreter> interpreter;

cv::Mat getTensorMat(int tnum, int debug);
void GetImageTFLite(float* out, cv::Mat &src);
void detect_from_video(cv::Mat &src);

struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

const RGB maskColor[2] = {{  0,   0,   0},  //background
                          {255, 255, 255}}; //person

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
	fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

cv::Mat getTensorMat(int tnum, int debug) {
    using namespace tflite;
    
	TfLiteType t_type = interpreter->tensor(tnum)->type;
	TFLITE_MINIMAL_CHECK(t_type == kTfLiteFloat32);

	TfLiteIntArray* dims = interpreter->tensor(tnum)->dims;
	if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
	TFLITE_MINIMAL_CHECK(dims->data[0] == 1);

	int h = dims->data[1];
	int w = dims->data[2];
	int c = dims->data[3];

	float* p_data = interpreter->typed_tensor<float>(tnum);
	TFLITE_MINIMAL_CHECK(p_data != nullptr);

	return cv::Mat(h,w,CV_32FC(c),p_data);
}

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
    Len=image.rows * image.cols * image.channels();
    for(i=0;i<Len;i++){
        f      = in[i];
        out[i] = f / 255.0f;
    }
}

void detect_from_video(cv::Mat &src)
{
    static cv::Mat image;
    static cv::Mat frame(model_width, model_height, CV_8UC3);
    static cv::Mat blend(src.cols   , src.rows    , CV_8UC3);

    GetImageTFLite(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    interpreter->Invoke();      // run your model

    // get max object per pixel
    //data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    cv::Mat output = getTensorMat(interpreter->outputs()[0], 0);
    cv::Mat ofinal(output.rows, output.cols,CV_8UC1);
    float* tmp = (float*)output.data;
    uint8_t* out = (uint8_t*)ofinal.data;

    for (unsigned int n = 0; n < output.total(); n++) {
			float exp0 = expf(tmp[2*n  ]);
			float exp1 = expf(tmp[2*n+1]);
			float p0 = exp0 / (exp0+exp1);
			float p1 = exp1 / (exp0+exp1);
			uint8_t val = (p0 < p1 ? 0 : 255);
			out[n] = (val & 0xE0) | (out[n] >> 3);
    }

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::Mat tmpbuf;
    //cv::dilate(ofinal,tmpbuf,element);
    //cv::erode(tmpbuf,ofinal,element);
    cv::morphologyEx(ofinal, tmpbuf, cv::MORPH_OPEN, element);
    cv::morphologyEx(tmpbuf, ofinal, cv::MORPH_CLOSE, element);

    // scale up into full-sized mask
    cv::Mat refined;
    cv::resize(ofinal, blend, cv::Size(src.cols, src.rows), cv::INTER_CUBIC);
    cv::GaussianBlur(blend, refined, cv::Size(0, 0), 3);
    cv::Mat bgr[3] = {refined, refined, refined};

    cv::Mat merged_mask;
    cv::merge(bgr, 3, merged_mask);
    cv::addWeighted(src, 0.5, merged_mask, 0.5, 0.0, src);
}

int main(){
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("./models/segm_full_v679.tflite");

    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    //https://github.com/google/mediapipe/tree/master/mediapipe/util/tflite/operations
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()); 
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
	outputVideo.open("/mnt/e/Data/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, outSize, true);

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