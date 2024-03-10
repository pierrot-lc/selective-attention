{
  description = "PyTorch devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    python-packages = ps:
      with ps; [
        pip
        setuptools
        virtualenv
      ];

    fhs = pkgs.buildFHSUserEnv {
      name = "pytorch";
      targetPkgs = pkgs: (with pkgs; [
        (python311.withPackages python-packages)
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
        just
        pdm
        zlib  # Required by numpy.
      ]);
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}
