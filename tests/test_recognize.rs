//
// test_recognize.rs
// Copyright (C) 2018 gmg137 <gmg137@live.com>
// Distributed under terms of the MIT license.
//

use hyper_lpr::*;

#[test]
fn test_recognize() {
    let pr = PipelinePR::new(
        "model/cascade.xml",
        "model/HorizonalFinemapping.prototxt",
        "model/HorizonalFinemapping.caffemodel",
        "model/Segmentation.prototxt",
        "model/Segmentation.caffemodel",
        "model/CharacterRecognization.prototxt",
        "model/CharacterRecognization.caffemodel",
        "model/SegmenationFree-Inception.prototxt",
        "model/SegmenationFree-Inception.caffemodel",
    );
    let pl = pr.plate_image_recognize("tests/demo.png", SEGMENTATION_FREE_METHOD, 36, 700);
    assert_eq!(pl[0].plate, "豫G8Y698");
}
