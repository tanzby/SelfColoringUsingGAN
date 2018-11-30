//
// Created by iceytan on 18-11-30.
//

#include <utils.h>

using namespace std;

void pbar(int cur, int total) {
    static const char *lable = "|/-\\";
    static char bar[52] = {0};
    static std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_cur = std::chrono::steady_clock::now();
    double t_delta = std::chrono::duration<double, std::milli>(t_cur - t_last).count();
    t_last = t_cur;
    string msg;
    int leftTimeMS = t_delta * (total - cur);
    int leftTimeSecond = leftTimeMS / 1000;
    leftTimeMS %= 1000;
    int leftTimeMinu = leftTimeSecond / 60;
    leftTimeSecond %= 60;
    int leftTimeHour = leftTimeMinu / 60;
    leftTimeMinu %= 60;
    int leftTimeDay = leftTimeHour / 24;
    leftTimeHour %= 24;
    if (leftTimeDay > 0) msg += to_string(leftTimeDay) + "d ";
    if (leftTimeHour > 0) msg += to_string(leftTimeHour) + "h ";
    if (leftTimeMinu > 0) msg += to_string(leftTimeMinu) + "m ";
    if (leftTimeSecond > 0) msg += to_string(leftTimeSecond) + "s ";
    if (leftTimeMinu == 0 && leftTimeMS > 0) msg += to_string(leftTimeMS) + "ms";
    printf(" [%-50s][%.2f%%][%s][%c]\r", bar, cur * 100.0 / total, msg.c_str(), lable[cur % 4]);
    std::fill(bar, bar + int(cur * 50.0 / total), '#');
    fflush(stdout);
}


pair<Context, Device> CreateGPUContextDevice() {
    std::vector<Platform> platforms;
    Platform::get(&platforms);
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties) (platforms[0])(),
                                          0};
    Context context(CL_DEVICE_TYPE_GPU, properties);
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cout << "Using " << devices[0].getInfo<CL_DEVICE_NAME>() << " for computing" << endl;
    return {context, devices[0]};
}