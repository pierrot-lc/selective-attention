{pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
} }:
(pkgs.buildFHSUserEnv {
  name = "pytorch";
  targetPkgs = pkgs: (with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    just
    python311
    python311Packages.pip
    python311Packages.virtualenv
    wget
    zlib
  ]);
  runScript = "bash";
}).env
