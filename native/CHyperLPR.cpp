/*
 * CHyperLPR.cpp
 * Copyright (C) 2018 gmg137 <gmg137@live.com>
 *
 * Distributed under terms of the GPLv3 license.
 */

#include "Pipeline.h"

using namespace std;

#ifdef __cplusplus
extern "C"
{
#endif

    // 创建车牌识别
    void* pr_pipeline_new(const char* c_detector_filename,
                         const char* c_finemapping_prototxt,const char* c_finemapping_caffemodel,
                         const char* c_segmentation_prototxt,const char* c_segmentation_caffemodel,
                         const char* c_charRecognization_proto,const char* c_charRecognization_caffemodel,
                         const char* c_segmentationfree_proto,const char* c_segmentationfree_caffemodel
            ){
        std::string detector_filename(c_detector_filename);
        std::string finemapping_prototxt(c_finemapping_prototxt);
        std::string finemapping_caffemodel(c_finemapping_caffemodel);
        std::string segmentation_prototxt(c_segmentation_prototxt);
        std::string segmentation_caffemodel(c_segmentation_caffemodel);
        std::string charRecognization_proto(c_charRecognization_proto);
        std::string charRecognization_caffemodel(c_charRecognization_caffemodel);
        std::string segmentationfree_proto(c_segmentationfree_proto);
        std::string segmentationfree_caffemodel(c_segmentationfree_caffemodel);
        return new pr::PipelinePR(detector_filename,
                                  finemapping_prototxt,
                                  finemapping_caffemodel,
                                  segmentation_prototxt,
                                  segmentation_caffemodel,
                                  charRecognization_proto,
                                  charRecognization_caffemodel,
                                  segmentationfree_proto,
                                  segmentationfree_caffemodel
                );
    }

    //const char* run_pipline_as_image(pr::PipelinePR* prc, cv::Mat* plateImage, int method){
        //std::vector<pr::PlateInfo> res = prc->RunPiplineAsImage(*plateImage, method);
        //const std::vector<std::string> CH_PLATE_CODE{"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"};

        //for(auto st:res) {
            //if(st.confidence>0.75 && st.getPlateName().length()==9) {
                //for(auto ch:CH_PLATE_CODE){
                    //if(ch==st.getPlateName().substr(0,3)){
                        //const char* lpr = st.getPlateName().c_str();
                        //return lpr;
                    //}
                //}
            //}
        //}
        //return "";
    //}

    // 从 mat 检测车牌,返回结果集
    void* plate_recognize(pr::PipelinePR* prc, cv::Mat* plateImage, int method){
        vector<pr::PlateInfo> *res = new vector<pr::PlateInfo>();
        *res = prc->RunPiplineAsImage(*plateImage, method);
        return res;
    }

    // 从图片检测车牌,返回结果集
    void* plate_recognize_as_image(pr::PipelinePR* prc, const char* imagePath, int method){
        cv::Mat src = cv::imread(std::string(imagePath));
        vector<pr::PlateInfo> *res = new vector<pr::PlateInfo>();
        *res = prc->RunPiplineAsImage(src, method);
        return res;
    }

    // 返回识别到的车牌数
    int get_plate_num(vector<pr::PlateInfo>* plateVec){
        return plateVec->size();
    }

    // 返回识别到的车牌对象
    void* get_plate(vector<pr::PlateInfo>* plateVec, int index){
        return &plateVec->at(index);
    }

    // 返回识别到的车牌号
    const char* get_plate_string(pr::PlateInfo* plate){
        const std::vector<std::string> CH_PLATE_CODE {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"};
        for(auto ch:CH_PLATE_CODE){
            if(ch==plate->getPlateName().substr(0,3)){
                const char* lpr = plate->getPlateName().c_str();
                return lpr;
            }
        }
        return "";
    }

    // 返回识别到的车牌图像
    void* get_plate_image(pr::PlateInfo* plate){
        cv::Mat* mat = new cv::Mat();
        *mat = plate->getPlateImage();
        return (mat);
    }

    // 返回识别准确度得分
    float get_plate_score(pr::PlateInfo* plate){
        return plate->confidence;
    }

    // 清除对象
    void pr_pipeline_drop(pr::PipelinePR* pr) {
        delete pr;
        pr = nullptr;
    }

#ifdef __cplusplus
}
#endif
