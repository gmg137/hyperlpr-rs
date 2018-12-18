//
// build.rs
// Copyright (C) 2018 gmg137 <gmg137@live.com>
// Distributed under terms of the MIT license.
//

fn main() {
    cc::Build::new()
        .cpp(true)
        .files(get_files("native"))
        .warnings(false)
        .extra_warnings(false)
        .compile("libhyperlpr.a");

    println!("cargo:rustc-link-lib=opencv_dnn");
}

fn get_files(path: &str) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(path)
        .unwrap()
        .into_iter()
        .filter_map(|x| x.ok().map(|x| x.path()))
        .filter(|x| x.extension().map(|e| e == "cpp").unwrap_or(false))
        .collect::<Vec<_>>()
}
