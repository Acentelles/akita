use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=AKITA_STAGE2_NATIVE_LIB_DIR");
    if let Some(dir) = env::var_os("AKITA_STAGE2_NATIVE_LIB_DIR") {
        assert_eq!(
            env::var("TARGET").unwrap(),
            "aarch64-apple-darwin",
            "native Stage2 target"
        );
        let dir = PathBuf::from(dir);
        assert!(dir.is_absolute(), "native library path must be absolute");
        let archive = dir.join("libakita_stage2_owned.a");
        assert!(
            archive.is_file(),
            "approved native archive must already exist"
        );
        println!("cargo:rerun-if-changed={}", archive.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=akita_stage2_owned");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }
}
