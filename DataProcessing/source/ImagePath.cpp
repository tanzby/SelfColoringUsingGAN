//
// Created by iceytan on 18-11-30.
//

#include <ImagePath.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

void mkdir(const string &path) {
    if (nullptr == opendir(path.c_str())) {
        mkdir(path.c_str(), 0775);
    }
}

std::vector<ImagePath> GetImagePath(
        const string &root_path,       // /path/to/data   e.g: /home/project/data
        const string &mask_path_name,       // mask images folder name e.g: inner_mask
        const string &background_path_name,
        const string &user_design_path_name,
        const string &save_path_name) {

    // ../data/inner_mask/test/*.png
    assert(nullptr != opendir((root_path + "/" + mask_path_name).c_str()));
    assert(nullptr != opendir((root_path + "/" + background_path_name).c_str()));
    assert(nullptr != opendir((root_path + "/" + user_design_path_name).c_str()));

    mkdir(root_path + "/" + save_path_name);
    mkdir(root_path + "/" + save_path_name + "/test");
    mkdir(root_path + "/" + save_path_name + "/train");

    std::vector<cv::String> mask_image_paths;
    cv::glob(root_path + "/" + mask_path_name + "/*.png", mask_image_paths, true);
    std::vector<ImagePath> res(mask_image_paths.size());


    auto modefy_point = mask_image_paths[0].find(mask_path_name);
    for (int i = 0; i < res.size(); ++i) {
        res[i].mask = mask_image_paths[i];
        res[i].origin = string(mask_image_paths[i]).replace(modefy_point, background_path_name.size(),
                                                            background_path_name);
        res[i].user = string(mask_image_paths[i]).replace(modefy_point, user_design_path_name.size(),
                                                          user_design_path_name);
    }

    return res;
}