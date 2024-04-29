{ rustc, cmake, openssl, pkg-config, buildPythonPackage, fetchFromGitHub, lib, rustPlatform }:

buildPythonPackage rec {
    pname = "uv";
    version = "0.1.35";
    pyproject = true;
    dontUseCmakeConfigure = true;

    OPENSSL_NO_VENDOR = 1;
    PKG_CONFIG_PATH="${openssl.dev}/lib/pkgconfig";

    src = fetchFromGitHub {
      owner = "astral-sh";
      repo = pname;
      rev = "${version}";
      hash = "sha256-GcAvpX7oanJ8G1dgTyTa8jk9xhTroF2G+ir8j7Yua1M=";
    };

    cargoDeps = rustPlatform.importCargoLock {
      lockFile = "${src}/Cargo.lock";
      outputHashes = {
        "async_zip-0.0.17" = "sha256-Q5fMDJrQtob54CTII3+SXHeozy5S5s3iLOzntevdGOs=";
        "pubgrub-0.2.1" = "sha256-sqC7R2mtqymYFULDW0wSbM/MKCZc8rP7Yy/gaQpjYEI=";
      };
    };

    nativeBuildInputs = with rustPlatform; [ pkg-config rustc cmake openssl.dev cargoSetupHook maturinBuildHook ];
}