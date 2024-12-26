//
//  ortwrapper.hpp
//
//  Created by zhaode on 2024/10/09.
//  ZhaodeWang
//

#ifndef ORTWRAPPER_hpp
#define ORTWRAPPER_hpp

#include <memory>
#include <nncase/runtime/interpreter.h>

namespace Ort {

class RuntimeManager {
public:
  RuntimeManager() {}

private:
};

class Module {
public:
  Module(std::shared_ptr<RuntimeManager> runtime, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    interpreter_.load_model(ifs).unwrap_or_throw();
    entry_function_ = interpreter_.entry_function().unwrap_or_throw();
  }

  nncase::tuple onForward(std::vector<nncase::value_t> &inputs) {
    return entry_function_->invoke(inputs)
        .unwrap_or_throw()
        .as<nncase::tuple>()
        .unwrap_or_throw();
  }

private:
  nncase::runtime::interpreter interpreter_;
  nncase::runtime::runtime_function *entry_function_;
};

template <typename T>
static nncase::tensor _Input(const std::vector<int> &shape,
                              std::shared_ptr<RuntimeManager> rtmgr) {
  nncase::dims_t shape_int64(shape.begin(), shape.end());
  return nncase::runtime::hrt::create(nncase::dt_float32, shape_int64)
      .unwrap_or_throw()
      .impl();
}

} // namespace Ort

#endif /* ORTWRAPPER_hpp */