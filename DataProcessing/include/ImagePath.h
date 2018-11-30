//
// Created by iceytan on 18-11-30.
//

#ifndef SOURCE_GENERATEIMG_IMAGEPATH_H
#define SOURCE_GENERATEIMG_IMAGEPATH_H

#include <string>
#include <vector>

using namespace std;

struct ImagePath {
    string mask;
    string user;
    string origin;
};


std::vector<ImagePath> GetImagePath(
        const string &root_path,       // /path/to/data   e.g: /home/project/data
        const string &mask_path_name,       // mask images folder name e.g: inner_mask
        const string &background_path_name,
        const string &user_design_path_name,
        const string &save_path_name);

#endif //SOURCE_GENERATEIMG_IMAGEPATH_H
