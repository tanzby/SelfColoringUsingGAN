#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <opencv2/opencv.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_SILENCE_DEPRECATION
#include <cl2.hpp>
using namespace std;
using namespace cl;


// 初始化OpenCL环境
pair<Context,Device> createGPUContextDevice(){
    std::vector<Platform> platforms;
    Platform::get(&platforms);
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM,
                                           (cl_context_properties)(platforms[0])(),
                                           0 };
    Context context(CL_DEVICE_TYPE_GPU, properties);
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cout <<"Using "<<devices[0].getInfo<CL_DEVICE_NAME>() << " for computing"<<  endl;
    return {context,devices[0]};
}

// 程序的内核函数
cl::Kernel createConvertKernel(const Context &context){

    static std::vector<string> programStrings {R"cl(
    __kernel void convert(
        read_only image2d_t usr,
        read_only image2d_t ori,
        write_only image2d_t dest,
        __global uchar* mas,
        const uint4 new_sky_color,
        const uint4 new_ground_color
    )
    {
        const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        int2 pos = (int2)(get_global_id(0), get_global_id(1));

        uchar mask = mas[pos.x+pos.y*get_global_size(0)];

        uint4 usr_pix = read_imageui(usr,sampler,pos);
        uint4 ori_pix = read_imageui(ori,sampler,pos);
        uint4 out_pix = ori_pix;

        if(mask == 255){
            if(all(usr_pix == (uint4)(153, 217, 234, 255))){
                out_pix = new_sky_color;
            }
            else if(all(usr_pix == (uint4)(181, 230, 29, 255)))
                out_pix = new_ground_color;
        }

        write_imageui(dest, pos, out_pix);
    })cl"};

    Program convertProgram (context,programStrings,CL_FALSE);
    try {
        convertProgram.build();
    }
    catch (...) {
        // Print build info for all devices
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = convertProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            cerr << pair.second << endl << endl;
        }
        exit(-1);
    }
    cout <<"Build program successfully." << endl;

    cl_int err;
    cl::Kernel kernel(convertProgram, "convert", &err);
    if(err!=CL_SUCCESS){
        cerr << "Get convert kernel error." << endl;
        exit(-1);
    }

    return kernel;
}

// 获取随机背景色
pair<cl_uint4,cl_uint4> getRandomTwoColor(){
    static std::vector<cl_uint4> colorTable {
        {153, 217, 234, 255},
        {181, 230, 29,  255},
        {128, 255, 215, 255},
        {237, 28,  36,  255},
        {255, 127, 39,  255},
        {255, 242, 0,   255},
        {185, 122, 87,  255},
        {163, 73,  164, 255},
        {255, 174, 201, 255},
        {30,  30,  30,  255},
        {127, 127, 127, 255}
    };
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(colorTable.begin(),colorTable.end(),g);
    return {colorTable[0],colorTable[1]};
}

struct ImageFiles{
    string user;
    string fore;
    string mask;
};

// 获取所有文件
std::vector<ImageFiles> getAllImageFiles(){
    std::vector<cv::String> filespath;
    cv::glob("data/bg_by_user/*.png",filespath,true);
    std::vector<ImageFiles> allfiles(filespath.size());
    for(int i = 0; i < filespath.size(); ++i){
        string fore(filespath[i]);
        string mask(filespath[i]);
        fore.replace(5,10,"foreground");
        mask.replace(5,10,"inner_mask");
        allfiles[i] = {filespath[i], fore, mask};
    }
    return allfiles;
}

// 显示进度条
void pbar(int cur, int total){
    static const char *lable = "|/-\\";
    static char bar[52]={0};
    static std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_cur = std::chrono::steady_clock::now();
    double t_delta =  std::chrono::duration<double, std::milli>(t_cur-t_last).count();
    t_last = t_cur;
    string msg;

    int leftTimeMS= t_delta* (total-cur);
    int leftTimeSecond= leftTimeMS/1000;leftTimeMS%=1000;
    int leftTimeMinu = leftTimeSecond/60;leftTimeSecond%=60;
    int leftTimeHour = leftTimeMinu/60;leftTimeMinu%=60;
    int leftTimeDay = leftTimeHour/24;leftTimeHour%=24;
    if (leftTimeDay>0){
        msg+= to_string(leftTimeDay)+"d ";
    }
    if (leftTimeHour>0){
        msg+= to_string(leftTimeHour)+"h ";
    }
    if (leftTimeMinu >0){
        msg+= to_string(leftTimeMinu)+"m ";
    }
    if (leftTimeSecond >0){
        msg+= to_string(leftTimeSecond)+"s ";
    }
    if (leftTimeMinu == 0 && leftTimeMS >0){
        msg+= to_string(leftTimeMS)+"ms";
    }
    printf(" [%-50s][%.2f%%][%s][%c]\r", bar, cur * 100.0/ total, msg.c_str() ,lable[cur%4]);
    for(int j = 0; j<= int(cur*50.0/total) ; j++)
    {
        bar[j] = '#';
    }
    fflush(stdout);
}


int main(){

    /* 环境初始化 */
    auto ctx_device  = createGPUContextDevice();
    Context context = ctx_device.first;
    Device  device  = ctx_device.second;
    CommandQueue commandQueue(context,device);
    auto convertKernel = createConvertKernel(context);
    int imageCols = 768;
    int imageRows = 768;
    int imageChannel = 4;
    ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
    const cl::array<size_type, 3>origin = { 0, 0, 0 };
    const cl::array<size_type, 3>region = { size_t(imageCols), size_t(imageRows), 1 };

    /* 声明缓冲对象 */
    cl_mem_flags img2d_in_flags = CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR;
    cl_mem_flags img2d_out_flags = CL_MEM_WRITE_ONLY;
    Image2D oriBuffer(context,img2d_in_flags,imageFormat,imageCols,imageRows);
    Image2D usrBuffer(context,img2d_in_flags,imageFormat,imageCols,imageRows);
    Image2D desBuffer(context,img2d_out_flags,imageFormat,imageCols,imageRows);
    Buffer maskBuffer(context,CL_MEM_READ_ONLY,imageCols*imageRows*sizeof(unsigned char));
    cv::Mat rgbUsrM, rgbOriM;


    std::vector<ImageFiles> data =  getAllImageFiles();
    for(int j = 0 ; j < data.size(); ++j){
        auto& i = data[j];
        string saveName;
        pbar(j,data.size());

        /* 读取图片 */
        cv::Mat masM = cv::imread(i.mask,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat usrM = cv::imread(i.user,CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat oriM = cv::imread(i.fore,CV_LOAD_IMAGE_UNCHANGED);
        cv::cvtColor(oriM, rgbOriM, CV_BGR2RGBA);
        cv::cvtColor(usrM, rgbUsrM, CV_BGR2RGBA);

        try
        {
            /* 复制图像的数据到GPU */
            commandQueue.enqueueWriteImage(oriBuffer, CL_TRUE, origin, region, 0, 0,  rgbOriM.data);
            commandQueue.enqueueWriteImage(usrBuffer, CL_TRUE, origin, region, 0, 0,  rgbUsrM.data);

            /* 复制数据到 mask buffer */
            commandQueue.enqueueWriteBuffer(maskBuffer,CL_TRUE,0,imageCols*imageRows*sizeof(uchar), masM.data);

            std::vector<std::pair<cl_uint4,cl_uint4>>  colorList{
                    {{153, 217,234,255},{181, 230, 29, 255}}
            };
            for (int k = 0; k < 3; ++k) {
                auto colors = getRandomTwoColor();
                colorList.emplace_back(colors.first,colors.second);
            }

            /* 一变四 */
            for (int k = 0; k < 4; ++k) {

                cl_uint4* newSkyColor = &colorList[k].first;
                cl_uint4* newGroundColor = &colorList[k].second;

                convertKernel.setArg(0, usrBuffer);
                convertKernel.setArg(1, oriBuffer);
                convertKernel.setArg(2, desBuffer);
                convertKernel.setArg(3, maskBuffer);
                convertKernel.setArg(4, *newSkyColor);
                convertKernel.setArg(5, *newGroundColor);

                /* 执行 kernel */
                commandQueue.enqueueNDRangeKernel(
                        convertKernel,
                        cl::NullRange,
                        NDRange(imageCols,imageRows),
                        cl::NullRange);

                /* 从GPU中复制数据 */
                commandQueue.enqueueReadImage(desBuffer,CL_TRUE,origin,region,0,0, rgbUsrM.data);

                /* 写图片 */
                char _ans[8];
                sprintf(_ans,"-%d.png",k);
                saveName = i.user;
                saveName.replace(5,10,"generate");
                saveName = saveName.substr(0,saveName.length()-4)+_ans;
                cv::cvtColor(rgbUsrM, rgbUsrM, CV_BGR2RGBA);
                cv::imwrite(saveName,rgbUsrM);
            }
        }
        catch (Error &e){
            cerr<< e.what() <<" "<< e.err()<< endl;
        }
        catch (cv::Exception & e){
            cerr << "\nWhile writing " << saveName << endl;
            cerr<< e.what() <<" "<< e.err << endl;
        }
    }
}