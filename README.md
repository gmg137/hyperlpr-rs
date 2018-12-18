#### [HyperLPR 高性能开源中文车牌识别框架](https://github.com/zeusees/HyperLPR) 的 Rust 绑定

#### 简单使用
```rust
    use hyper_lpr;

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
    let pl = pr.plate_image_recognize("demo.png", SEGMENTATION_FREE_METHOD, 36, 700);
    pl.iter().for_each(|p|{
        println!("plate: {}",p.plate);
    });
```
