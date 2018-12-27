//
// hyperlpr.rs
// Copyright (C) 2018 gmg137 <gmg137@live.com>
// Distributed under terms of the MIT license.
//
//! #### [HyperLPR 高性能开源中文车牌识别框架](https://github.com/zeusees/HyperLPR) 的 Rust 绑定
//!
//! #### Example
//! ```rust
//!     use hyper_lpr::*;
//!
//!     let pr = PipelinePR::new(
//!         "model/cascade.xml",
//!         "model/HorizonalFinemapping.prototxt",
//!         "model/HorizonalFinemapping.caffemodel",
//!         "model/Segmentation.prototxt",
//!         "model/Segmentation.caffemodel",
//!         "model/CharacterRecognization.prototxt",
//!         "model/CharacterRecognization.caffemodel",
//!         "model/SegmenationFree-Inception.prototxt",
//!         "model/SegmenationFree-Inception.caffemodel",
//!     );
//!     let pl = pr.plate_image_recognize("tests/demo.png", SEGMENTATION_FREE_METHOD, 36, 700);
//!     pl.iter().for_each(|p|{
//!         println!("plate: {}",p.plate);
//!     });
//!     assert_eq!(pl.len(), 1);
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

/// 图片切割方法
pub const SEGMENTATION_FREE_METHOD: c_int = 0;
/// 图片切割方法
pub const SEGMENTATION_BASED_METHOD: c_int = 1;

#[repr(C)]
struct CPipelinePR {
    _private: [u8; 0],
}
#[repr(C)]
struct CVecPlateInfo {
    _private: [u8; 0],
}
#[repr(C)]
struct CPlateInfo {
    _private: [u8; 0],
}

/// 识别到的车牌信息
#[derive(Debug)]
pub struct PlateInfo {
    inner: *const CPlateInfo,
    /// 车牌图像
    pub image: cv::Mat,
    /// 车牌号
    pub plate: String,
}
/// 车牌识别
#[derive(Debug)]
pub struct PipelinePR {
    inner: *mut CPipelinePR,
}

unsafe impl Send for CPipelinePR {}
unsafe impl Send for PipelinePR {}
unsafe impl Send for CPlateInfo {}
unsafe impl Send for PlateInfo {}

extern "C" {
    fn pr_pipeline_new(
        c_detector_filename: *const c_char,
        c_finemapping_prototxt: *const c_char,
        c_finemapping_caffemodel: *const c_char,
        c_segmentation_prototxt: *const c_char,
        c_segmentation_caffemodel: *const c_char,
        c_charRecognization_proto: *const c_char,
        c_charRecognization_caffemodel: *const c_char,
        c_segmentationfree_proto: *const c_char,
        c_segmentationfree_caffemodel: *const c_char,
    ) -> *mut CPipelinePR;
    fn plate_recognize(
        prc: *const CPipelinePR,
        plate_image: *const cv::CMat,
        method: c_int,
        min_w: c_int,
        max_w: c_int,
    ) -> *const CVecPlateInfo;
    fn plate_recognize_as_image(
        prc: *const CPipelinePR,
        image_path: *const c_char,
        method: c_int,
        min_w: c_int,
        max_w: c_int,
    ) -> *const CVecPlateInfo;
    fn get_plate_num(vpi: *const CVecPlateInfo) -> c_int;
    fn get_plate(vpi: *const CVecPlateInfo, index: c_int) -> *const CPlateInfo;
    fn get_plate_string(plate_info: *const CPlateInfo) -> *const c_char;
    fn get_plate_image(plate_info: *const CPlateInfo) -> *mut cv::CMat;
    fn get_plate_score(plate_info: *const CPlateInfo) -> c_float;
    fn pr_pipeline_drop(prc: *const CPipelinePR);
}

impl PipelinePR {
    /// 初始化车牌识别，载入模型文件
    pub fn new(
        detector_filename: &str,
        finemapping_prototxt: &str,
        finemapping_caffemodel: &str,
        segmentation_prototxt: &str,
        segmentation_caffemodel: &str,
        charrecognization_proto: &str,
        charrecognization_caffemodel: &str,
        segmentationfree_proto: &str,
        segmentationfree_caffemodel: &str,
    ) -> Self {
        let c_detector_filename = CString::new(detector_filename).unwrap();
        let c_finemapping_prototxt = CString::new(finemapping_prototxt).unwrap();
        let c_finemapping_caffemodel = CString::new(finemapping_caffemodel).unwrap();
        let c_segmentation_prototxt = CString::new(segmentation_prototxt).unwrap();
        let c_segmentation_caffemodel = CString::new(segmentation_caffemodel).unwrap();
        let c_charrecognization_proto = CString::new(charrecognization_proto).unwrap();
        let c_charrecognization_caffemodel = CString::new(charrecognization_caffemodel).unwrap();
        let c_segmentationfree_proto = CString::new(segmentationfree_proto).unwrap();
        let c_segmentationfree_caffemodel = CString::new(segmentationfree_caffemodel).unwrap();
        let cap = unsafe {
            pr_pipeline_new(
                c_detector_filename.as_ptr(),
                c_finemapping_prototxt.as_ptr(),
                c_finemapping_caffemodel.as_ptr(),
                c_segmentation_prototxt.as_ptr(),
                c_segmentation_caffemodel.as_ptr(),
                c_charrecognization_proto.as_ptr(),
                c_charrecognization_caffemodel.as_ptr(),
                c_segmentationfree_proto.as_ptr(),
                c_segmentationfree_caffemodel.as_ptr(),
            )
        };
        PipelinePR { inner: cap }
    }

    /// 从图片检测车牌,返回结果集
    ///
    /// - image_path 图片路径
    /// - model 图片切割方法: SEGMENTATION_FREE_METHOD, SEGMENTATION_BASED_METHOD
    /// - min_w 要识别的车牌框最小宽度
    /// - max_w 要识别的车牌框最大宽度
    ///
    /// 注: 在视频采集时 min_w 与 max_w 对性能影响极大，建议根据视频图像大小和范围调整
    pub fn plate_image_recognize(
        &self,
        image_path: &str,
        model: c_int,
        min_w: u32,
        max_w: u32,
    ) -> Vec<PlateInfo> {
        let path = CString::new(image_path).unwrap();
        let mut result: Vec<PlateInfo> = Vec::new();
        unsafe {
            let vp = plate_recognize_as_image(
                self.inner,
                path.as_ptr(),
                model,
                min_w as c_int,
                max_w as c_int,
            );
            let num = get_plate_num(vp) as u32;
            for i in 0..num {
                let cp = get_plate(vp, i as c_int);
                let p = CStr::from_ptr(get_plate_string(cp))
                    .to_string_lossy()
                    .into_owned();
                if p.len() >= 9 && get_plate_score(cp) as f64 > 0.77 {
                    result.push(PlateInfo {
                        inner: cp,
                        image: cv::Mat::from_raw(get_plate_image(cp)),
                        plate: p,
                    });
                }
            }
        }
        return result;
    }

    /// 从 mat 检测车牌,返回结果集
    ///
    /// - mat openCV 图像
    /// - model 图片切割方法: SEGMENTATION_FREE_METHOD, SEGMENTATION_BASED_METHOD
    /// - min_w 要识别的车牌框最小宽度
    /// - max_w 要识别的车牌框最大宽度
    ///
    /// 注: 在视频采集时 min_w 与 max_w 对性能影响极大，建议根据视频图像大小和范围调整
    pub fn plate_recognize(
        &self,
        mat: &cv::Mat,
        model: c_int,
        min_w: u32,
        max_w: u32,
    ) -> Vec<PlateInfo> {
        let mut result: Vec<PlateInfo> = Vec::new();
        unsafe {
            let vp = plate_recognize(self.inner, mat.inner, model, min_w as c_int, max_w as c_int);
            let num = get_plate_num(vp) as u32;
            for i in 0..num {
                let cp = get_plate(vp, i as c_int);
                let p = CStr::from_ptr(get_plate_string(cp))
                    .to_string_lossy()
                    .into_owned();
                if p.len() >= 9 && get_plate_score(cp) as f64 > 0.90 {
                    result.push(PlateInfo {
                        inner: cp,
                        image: cv::Mat::from_raw(get_plate_image(cp)),
                        plate: p,
                    });
                }
            }
        }
        return result;
    }
}

impl Drop for PipelinePR {
    fn drop(&mut self) {
        unsafe {
            pr_pipeline_drop(self.inner);
        }
    }
}
