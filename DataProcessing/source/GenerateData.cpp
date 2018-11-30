//
// Created by iceytan on 18-11-30.
//

#include <iostream>
#include <ImagePath.h>
#include <utils.h>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <spline.h>

namespace {

    using PixelType = cv::Vec3b;
    using PointType = cv::Point2i;


    const int IMG_W = 768;
    const int IMG_H = 768;
    const int IMG_C = 4;
    const ImageFormat IMG_FORMAT(CL_RGBA, CL_UNSIGNED_INT8);
    const cl::array<size_type, 3> origin = {0, 0, 0};
    const cl::array<size_type, 3> region = {size_t(IMG_W), size_t(IMG_H), 1};
    cl_mem_flags IMG_IN_FLAG = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
    cl_mem_flags IMG_OUT_FLAG = CL_MEM_WRITE_ONLY;

    const int GENERATE_NUM = 4;

    const int SAMPLE_RADIUS = 12;

    const int EXPECT_MAX_LEN = 16;

    const int EXPECT_RANGE = 8;

    const int LINE_WIDTH = 8;

    int CUR_MAX_LEN;

    int CUR_IMG_TYPE;

    cv::Mat mask_mat, user_mat, origin_mat;

    std::vector<std::vector<uchar>> pixel_info_vec;

    std::mt19937 rg = std::mt19937(std::random_device()());

    int GetRandomNumFromRange(const int &end, const int &start = 0) {
        assert(end > start);
        return rg() % (end - start) + start;
    }

    cl::Kernel CreateConvertKernel(const Context &context) {

        static std::vector<string> programStrings{R"cl(
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

        Program convertProgram(context, programStrings, CL_FALSE);
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
        cout << "Build program successfully." << endl;

        cl_int err;
        cl::Kernel kernel(convertProgram, "convert", &err);
        if (err != CL_SUCCESS) {
            cerr << "Get convert kernel error." << endl;
            exit(-1);
        }

        return kernel;
    }

    void GetPixelInfo(cv::Mat &input) {
        const int d_degree = 5;
        auto &sky_color = input.at<PixelType>(1, 1);
        auto &ground_color = input.at<PixelType>(IMG_W - 1, IMG_H - 1);

        for (int c = SAMPLE_RADIUS; c < mask_mat.cols; c += SAMPLE_RADIUS << 1) {
            for (int r = SAMPLE_RADIUS; r < mask_mat.rows; r += SAMPLE_RADIUS << 1) {
                int _y = (r - SAMPLE_RADIUS) / (SAMPLE_RADIUS << 1);
                int _x = (c - SAMPLE_RADIUS) / (SAMPLE_RADIUS << 1);

                // check pixel
                for (int _radius = 1; _radius <= SAMPLE_RADIUS; ++_radius) {
                    for (int d = 0; d <= 360; d += d_degree) {
                        int rx = c + cos(180.f * d / M_PI) * _radius, ry = r + sin(180.f * d / M_PI) * _radius;
                        if (rx < 0 || rx >= mask_mat.cols || ry < 0 || ry > mask_mat.rows) continue;
                        if (mask_mat.at<uchar>(ry, rx) < 10) {
                            pixel_info_vec[_y][_x] = 0;
                            _radius = SAMPLE_RADIUS + 1; // break the outside loop
                            break;
                        }
                    }
                }
                auto &_gc = input.at<PixelType>(r, c);
                if (_gc == sky_color) pixel_info_vec[_y][_x] = 1;
                else if (_gc == ground_color) pixel_info_vec[_y][_x] = 2;
                else pixel_info_vec[_y][_x] = 0;
            }
        }
    }

    std::vector<PointType> GetCurvelPoint(std::vector<PointType> input, int dx = 1) {
        std::vector<PointType> result;
        std::vector<double> X(input.size()), Y(input.size());
        for (int i = 0; i < input.size(); ++i) {
            X[i] = input[i].x;
            Y[i] = input[i].y;
        }
        tk::spline s;
        s.set_points(X, Y);    // currently it is required that X is already sorted
        for (int sx = input[0].x; sx <= input.back().x; sx += dx) {
            result.emplace_back(sx, s(sx));
        }
        return result;
    }

    PointType RandomSelect(std::vector<PointType> points) {
        assert(!points.empty());
        return points[GetRandomNumFromRange(points.size())];
    }

    void MarkPointInRange(const PointType &p) {
        const int &x = (p.x - SAMPLE_RADIUS) / (SAMPLE_RADIUS << 1);
        const int &y = (p.y - SAMPLE_RADIUS) / (SAMPLE_RADIUS << 1);

        int rangeX = 5;
        int rangeY = GetRandomNumFromRange(SAMPLE_RADIUS, SAMPLE_RADIUS / 2);

        for (int r = y - rangeY / 2; r <= y + rangeY / 2; ++r) {
            for (int c = x - rangeX / 2; c <= x + rangeX / 2; ++c) {
                if (c < 0 || r < 0 || r >= pixel_info_vec.size() || c >= pixel_info_vec[0].size()) {
                    continue;
                }
                pixel_info_vec[r][c] = 0;
            }
        }
    }


    void DFSDRAW(const PointType &p, std::vector<PointType> &curve_points) {

        curve_points.emplace_back((p.x * SAMPLE_RADIUS << 1) + SAMPLE_RADIUS,
                                  (p.y * SAMPLE_RADIUS << 1) + SAMPLE_RADIUS);
        if (curve_points.size() == CUR_MAX_LEN) return;


        static std::vector<int> dx{3};
        static std::vector<int> dy{-3, -2, -1, 0, 1, 2, 3};

        std::vector<PointType> _tmp_point_set;
        for (const auto &_dx: dx) {
            for (const auto &_dy: dy) {
                int cur_r = p.y + _dy;
                int cur_c = p.x + _dx;
                bool out_boundary = cur_c < 0 || cur_c >= pixel_info_vec[0].size()
                                    || cur_r < 0 || cur_r >= pixel_info_vec.size();

                if (out_boundary) continue;

                bool unreachable = false;
                for (int _x = p.x; _x <= cur_c && !unreachable; ++_x) {
                    if (_dy > 0) {
                        for (int _y = p.y; _y <= cur_r; ++_y) {
                            if (pixel_info_vec[_y][_x] != CUR_IMG_TYPE) {
                                unreachable = true;
                                break;
                            }
                        }
                    } else {
                        for (int _y = p.y; _y >= cur_r; --_y) {
                            if (pixel_info_vec[_y][_x] != CUR_IMG_TYPE) {
                                unreachable = true;
                                break;
                            }
                        }
                    }
                }

                if (unreachable || pixel_info_vec[cur_r][cur_c] != CUR_IMG_TYPE) continue;
                _tmp_point_set.emplace_back(cur_c, cur_r);
            }
        }
        if (_tmp_point_set.empty())
            return;

        auto select_p = RandomSelect(_tmp_point_set);
        DFSDRAW(select_p, curve_points);
    }

    void GenerateLinesPoints(std::vector<std::vector<PointType>> &sky_curve_points_sets,
                             std::vector<std::vector<PointType>> &ground_curve_points_sets, int max_curve_len) {
        CUR_MAX_LEN = max_curve_len;
        sky_curve_points_sets.clear();
        ground_curve_points_sets.clear();

        /// 对于每一列，随机选择一些点出发
        for (int c = 0; c < pixel_info_vec.size(); ++c) {
            std::vector<PointType> _select_;
            for (int r = 0; r < pixel_info_vec.size(); ++r) {
                if (pixel_info_vec[r][c] != 0) {
                    _select_.emplace_back(c, r);
                }
            }

            std::shuffle(_select_.begin(), _select_.end(), std::mt19937(std::random_device()()));

            for (auto &sp: _select_) {
                auto &pinfo = pixel_info_vec[sp.y][sp.x];
                if (pinfo != 0) {
                    CUR_IMG_TYPE = pinfo;
                    std::vector<PointType> curve_points;
                    DFSDRAW(sp, curve_points);
                    if (curve_points.size() >= 3) {
                        if (CUR_IMG_TYPE == 1) {
                            sky_curve_points_sets.push_back(curve_points);
                        } else {
                            ground_curve_points_sets.push_back(curve_points);
                        }

                        for (auto &p: curve_points) {
                            MarkPointInRange(p);
                        }
                    }
                }
            }
        }
    }

    void DrawLines(cv::Mat &input, std::vector<std::vector<PointType>> v,
                   const PixelType &color, const int &line_width = 2) {
        for (auto &curvePoints: v) {
            auto points = GetCurvelPoint(curvePoints);
            polylines(input, points, false, color, line_width, cv::LineTypes::LINE_AA);
        }
    }


    pair<cl_uint4, cl_uint4> GetRandomTwoColor() {
        static std::vector<cl_uint4> colorTable{
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
        std::shuffle(colorTable.begin(), colorTable.end(), rg);
        return {colorTable[0], colorTable[1]};
    }

    const std::pair<cl_uint4, cl_uint4> DEFAULT_COLOR{
            {153, 217, 234, 255},
            {181, 230, 29,  255}
    };


    PixelType cint2pt(const cl_uint4 &ci) {
        return {static_cast<uchar>(ci.z),
                static_cast<uchar>(ci.y),
                static_cast<uchar>(ci.x)};
    }


}


int main() {
    /// read origin data's path
    const string save_path_name = "pair";
    const string root_path = "../data";
    const string mask_path = "inner_mask";
    const string front_path = "foreground";
    const string colored_path = "bg_by_user";
    auto images = GetImagePath(root_path, mask_path, front_path, colored_path, save_path_name);

    /// init CL Env
    auto ctx_device = CreateGPUContextDevice();
    Context context = ctx_device.first;
    Device device = ctx_device.second;
    CommandQueue commandQueue(context, device);
    auto convertKernel = CreateConvertKernel(context);
    Image2D oriBuffer(context, IMG_IN_FLAG, IMG_FORMAT, IMG_W, IMG_H);
    Image2D usrBuffer(context, IMG_IN_FLAG, IMG_FORMAT, IMG_W, IMG_H);
    Image2D desBuffer(context, IMG_OUT_FLAG, IMG_FORMAT, IMG_W, IMG_H);
    Buffer maskBuffer(context, CL_MEM_READ_ONLY, IMG_W * IMG_H * sizeof(unsigned char));

    /// generate imgs

    int train_image_idx = 0;
    int test_image_idx = 0;
    string p_prefixname = root_path + "/" + save_path_name + "/";
    string _save_name;
    auto prefix_pos = images[0].mask.find(mask_path) + mask_path.length() + 2;

    const uint len = uint(768.f / (SAMPLE_RADIUS << 1) + 1);
    pixel_info_vec.resize(len);
    for (auto &p:pixel_info_vec) p.resize(len);


    for (int i = 0; i < images.size(); ++i) {
        // display process
        pbar(i, images.size());

        // get image path and read
        auto &IMG = images[i];
        mask_mat = cv::imread(IMG.mask, CV_LOAD_IMAGE_GRAYSCALE);
        user_mat = cv::imread(IMG.user, CV_LOAD_IMAGE_UNCHANGED);
        origin_mat = cv::imread(IMG.origin, CV_LOAD_IMAGE_UNCHANGED);

        // coloring other 4 image

        try {
            // convert the format of OpenCV' mat
            cv::Mat _tmp_user_mat, _tmp_origin_mat;
            cv::cvtColor(origin_mat, _tmp_origin_mat, CV_BGR2RGBA);
            cv::cvtColor(user_mat, _tmp_user_mat, CV_BGR2RGBA);

            commandQueue.enqueueWriteImage(oriBuffer, CL_TRUE, origin, region, 0, 0, _tmp_origin_mat.data);
            commandQueue.enqueueWriteImage(usrBuffer, CL_TRUE, origin, region, 0, 0, _tmp_user_mat.data);
            commandQueue.enqueueWriteBuffer(maskBuffer, CL_TRUE, 0, IMG_H * IMG_W * sizeof(uchar), mask_mat.data);

            std::vector<std::pair<cl_uint4, cl_uint4>> colorList{DEFAULT_COLOR};

            for (int k = 0; k < GENERATE_NUM - 1; ++k) {
                auto colors = GetRandomTwoColor();
                colorList.emplace_back(colors.first, colors.second);
            }

            for (int k = 0; k < GENERATE_NUM; ++k) {

                cl_uint4 *new_sky = &colorList[k].first;
                cl_uint4 *new_ground = &colorList[k].second;

                convertKernel.setArg(0, usrBuffer);
                convertKernel.setArg(1, oriBuffer);
                convertKernel.setArg(2, desBuffer);
                convertKernel.setArg(3, maskBuffer);
                convertKernel.setArg(4, *new_sky);
                convertKernel.setArg(5, *new_ground);
                commandQueue.enqueueNDRangeKernel(convertKernel, cl::NullRange, NDRange(IMG_W, IMG_H), cl::NullRange);
                commandQueue.enqueueReadImage(desBuffer, CL_TRUE, origin, region, 0, 0, _tmp_user_mat.data);

                cv::Mat target_image;
                cv::cvtColor(_tmp_user_mat, target_image, CV_BGR2RGB);


                /*
                 *  Now draw a line to background according to target_image and mask and save
                 *
                 *  color:      target_image
                 *  mask :      mask
                 *  background: origin_mat
                 */

                GetPixelInfo(target_image);


                std::vector<std::vector<PointType >> sky_curve_points_sets;
                std::vector<std::vector<PointType >> ground_curve_points_sets;
                GenerateLinesPoints(sky_curve_points_sets, ground_curve_points_sets,
                                    GetRandomNumFromRange(EXPECT_MAX_LEN + EXPECT_RANGE / 2,
                                                          EXPECT_MAX_LEN - EXPECT_RANGE / 2));


                cv::Mat input_image, pair_mat;
                origin_mat.copyTo(input_image);

                DrawLines(input_image, sky_curve_points_sets, cint2pt(*new_sky), LINE_WIDTH);
                DrawLines(input_image, ground_curve_points_sets, cint2pt(*new_ground), LINE_WIDTH);

                cv::hconcat(input_image, target_image, pair_mat);

                if (IMG.mask[prefix_pos] == 'r') {
                    _save_name = p_prefixname + "train/" + to_string(++train_image_idx) + ".png";
                } else {
                    _save_name = p_prefixname + "test/" + to_string(++test_image_idx) + ".png";
                }

                cv::resize(pair_mat, pair_mat, {512, 256});
                cv::imwrite(_save_name, pair_mat);

                // cv::imshow("pair",pair_mat);
                // cv::waitKey(0);

            }
        }
        catch (Error &e) {
            cerr << e.what() << " " << e.err() << endl;
        }
        catch (cv::Exception &e) {
            cerr << "\nWhile writing " << _save_name << endl;
            cerr << e.what() << " " << e.err << endl;
        }
    }
}