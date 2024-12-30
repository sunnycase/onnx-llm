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
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>
#include <type_traits>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace Ort {

class RuntimeManager {
public:
  RuntimeManager() {}

private:
};

class Module {
public:
  size_t count = 0;
  Module(std::shared_ptr<RuntimeManager> runtime, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    interpreter_.load_model(ifs).unwrap_or_throw();
    entry_function_ = interpreter_.entry_function().unwrap_or_throw();
  }
  void dump_input(std::ofstream &desc_file, nncase::value_t &input_data, std::string input_name, std::string dtype, size_t count)
  {
    auto tensor_ = input_data.as<nncase::tensor>().expect("not tensor");
    auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
    auto shape = tensor_->shape();
    auto datasize = 1;
    desc_file<< dtype << ": ";
    for(auto ii : shape)
    {
      desc_file<< ii << " ";
      datasize*=ii;
    }
    desc_file<<std::endl;

    std::ofstream oufile(input_name+std::to_string(count)+".bin", std::ios::binary);
    if (oufile) 
    {
      oufile.write(reinterpret_cast<char*>(data), datasize * sizeof(float));
      oufile.close();
    }
  }
  nncase::tuple onForward(std::vector<nncase::value_t> &inputs) {
    if (0)
    {
      fs::path dir_path = "calib";
      try {
          fs::create_directory(dir_path);
          std::cout << "Directory created successfully: " << dir_path << std::endl;
      } catch (const fs::filesystem_error& e) {
          std::cerr << "Error: " << e.what() << std::endl;
      }
      std::ofstream outputFile("calib/input_desc"+std::to_string(count)+".txt");  
      dump_input(outputFile, inputs[0], "calib/input_ids_float", "fp32", count);
      dump_input(outputFile, inputs[1], "calib/attention_mask_float", "fp32", count);
      dump_input(outputFile, inputs[2], "calib/postion_ids_int", "i32", count);
      dump_input(outputFile, inputs[3], "calib/past_key_values_float", "fp32", count);
      count+=1;
    }
  
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
  return nncase::runtime::hrt::create(
             std::is_same_v<T, float> ? nncase::dt_float32 : nncase::dt_int32,
             shape_int64, nncase::runtime::host_runtime_tensor::pool_shared)
      .unwrap_or_throw()
      .impl();
}

} // namespace Ort

#endif /* ORTWRAPPER_hpp */