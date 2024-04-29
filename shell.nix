let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11";
  pkgs = import nixpkgs { config = {allowUnfree = true;}; overlays = []; };
  nixpkgs_unstable = fetchTarball "https://github.com/NixOS/nixpkgs/archive/69ee1d82f1fa4c70a3dc9a64111e7eef3b8e4527.tar.gz";
  pkgs_unstable = import nixpkgs_unstable { config = {allowUnfree = true;}; overlays = []; };
  nixpkgs-python = import (fetchTarball "https://github.com/cachix/nixpkgs-python/archive/refs/heads/main.zip");
  python = nixpkgs-python.packages.x86_64-linux."3.12.1";
  uv = ps: (ps.callPackage ./uv.nix {rustc = pkgs_unstable.rustc;} );
  cuda_pkg = pkgs_unstable.cudaPackages.cudatoolkit;
  lib_pkgs = [ pkgs.linuxPackages.nvidia_x11 pkgs.stdenv.cc.cc.lib pkgs.zlib ];
in
pkgs.mkShell {

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath lib_pkgs;

  packages = [
    (python.withPackages(ps: with ps; [(uv ps)]))
    cuda_pkg
    pkgs.zlib
    pkgs.pkg-config
    pkgs.cairo
    pkgs.expat
    pkgs.xorg.libXdmcp
    pkgs.ninja
    pkgs.gobject-introspection
    pkgs.cmake
  ];
}
