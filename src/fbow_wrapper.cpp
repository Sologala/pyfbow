
#include <iostream>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// fbow
#include "fbow.h"
#include "string"
#include "vocabulary_creator.h"
#include <vector>
using namespace std;
#include "ndarray_converter.h"
using namespace pybind11::literals;

string version() { return "0.0.1"; }

class Vocabulary {
public:
  Vocabulary(int k = 10, int L = 6, int nthreads = 1, int maxIters = 0,
             bool verbose = true) {
    voc_creator_params.k = k;
    voc_creator_params.L = L;
    voc_creator_params.nthreads = nthreads;
    voc_creator_params.maxIters = maxIters;
    voc_creator_params.verbose = verbose;
    _verbose = verbose;
    voc = new fbow::Vocabulary();
    voc_creator = new fbow::VocabularyCreator();
  }

  ~Vocabulary() {
    if (_verbose)
      std::cout << "Entering destructor" << std::endl;
    delete voc;
    delete voc_creator;
    if (_verbose)
      std::cout << "Exiting destructor" << std::endl;
  }

  void create(cv::Mat &training_feat_vec) {
    std::srand(0);
    if (training_feat_vec.dims != 2) {
      cout << "input feature should be like [ N x dim_feature]";
      return;
    }
    int N = training_feat_vec.rows;
    int feature_dim = training_feat_vec.cols;
    std::vector<cv::Mat> vec(N);
    for (int i = 0, sz = N; i < sz; i++) {
      vec[i] = training_feat_vec.row(i);
    }
    cout << "fbow is Creating with [" << N << " " << feature_dim << "] feature"
         << endl;
    voc_creator->create(*voc, vec, std::string("orb"), voc_creator_params);
  }

  void clear() { voc->clear(); }

  void readFromFile(const std::string &path) {
    voc->readFromFile(path);
    if (_verbose) {
      cout << "desp bytes size :" << voc->getDescSize() << endl;
      cout << "node num: " << voc->size() << endl;
    }
  }

  void saveToFile(const std::string &path) { voc->saveToFile(path); }

  std::map<uint32_t, float> transform(const cv::Mat &training_feat_vec) {
    fbow::fBow word;
    std::map<uint32_t, float> ret;

    if (_verbose)
      cout << "input size " << training_feat_vec.rows << " "
           << training_feat_vec.cols << endl;
    word = voc->transform(training_feat_vec);
    for (auto p : word) {
      ret[p.first] = p.second;
    }
    return ret;
  }

  std::map<uint32_t, std::vector<uint32_t>>
  transform_with_feature(const cv::Mat &training_feat_vec, int level) {
    fbow::fBow word;
    fbow::fBow2 word_feature;
    std::map<uint32_t, std::vector<uint32_t>> ret;

    if (_verbose)
      cout << "input size " << training_feat_vec.rows << " "
           << training_feat_vec.cols << endl;
    voc->transform(training_feat_vec, level, word, word_feature);
    for (auto p : word_feature) {
      ret[p.first] = p.second;
    }
    return ret;
  }

  void temp(cv::Mat &mat) { std::cout << mat << std::endl; }

  fbow::Vocabulary *voc;
  fbow::VocabularyCreator *voc_creator;
  fbow::VocabularyCreator::Params voc_creator_params;
  bool _verbose;
};

PYBIND11_MODULE(pyfbow, m) {
  NDArrayConverter::init_numpy();
  m.doc() = "pybind11 of fbow"; // optional module docstring
  m.def("__version__", &version, "get the version of fbow");
  py::class_<Vocabulary>(m, "Vocabulary")
      .def(py::init<int, int, int, int, bool>(), "K"_a = 10, "L"_a = 6,
           "nthreads"_a = 1, "maxIters"_a = 0, "verbose"_a = true)
      //   .def("test", &Vocabulary::temp, "mat"_a)
      .def("create", &Vocabulary::create, "features"_a)
      .def("clear", &Vocabulary::clear)
      .def("readFromFile", &Vocabulary::readFromFile, "path"_a)
      .def("saveToFile", &Vocabulary::saveToFile, "path"_a)
      .def("transform", &Vocabulary::transform, "feature"_a)
      .def("transform_with_feature", &Vocabulary::transform_with_feature,
           "feature"_a, "level"_a)
      .def("clear", &Vocabulary::clear);
}