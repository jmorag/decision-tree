with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
  python37
  python37Packages.numpy
  python37Packages.scipy
  python37Packages.scikitlearn
  python37Packages.matplotlib
  ];
}
