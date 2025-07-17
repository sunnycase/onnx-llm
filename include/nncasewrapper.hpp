//
//  ortwrapper.hpp
//
//  Created by zhaode on 2024/10/09.
//  ZhaodeWang
//

#ifndef ORTWRAPPER_hpp
#define ORTWRAPPER_hpp

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nncase/llm/paged_attention_config.h>
#include <nncase/llm/paged_attention_scheduler.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stream.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/value.h>
#include <type_traits>
#include <vector>

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
        using namespace nncase;
        using namespace nncase::llm;
        using namespace nncase::runtime;

        std::ifstream ifs(path, std::ios::binary);
        nncase::runtime::std_istream stream(ifs);
        interpreter_.load_model(stream).unwrap_or_throw();
        entry_function_ = interpreter_.entry_function().unwrap_or_throw();
        paged_attention_config_ = paged_attention_config(
            std::in_place, 28, 8, 64, nncase::dt_float32, 256,
            std::array<paged_kvcache_dim_kind, 6>{
                paged_kvcache_dim_kind::num_blocks,
                paged_kvcache_dim_kind::num_layers, paged_kvcache_dim_kind::kv,
                paged_kvcache_dim_kind::num_kv_heads,
                paged_kvcache_dim_kind::head_dim,
                paged_kvcache_dim_kind::block_size},
            std::vector<paged_kvcache_dim_kind>{
                paged_kvcache_dim_kind::head_dim},
            dims_t{32},
            std::vector<paged_kvcache_dim_kind>{
                paged_kvcache_dim_kind::num_blocks},
            std::vector<dims_t>{dims_t{0}});
    }

    void dump_input(std::ofstream &desc_file, nncase::value_t &input_data,
                    std::string input_name, std::string dtype, size_t count) {
        auto tensor_ = input_data.as<nncase::tensor>().expect("not tensor");
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape = tensor_->shape();
        auto datasize = 1;
        desc_file << dtype << ": ";
        for (auto ii : shape) {
            desc_file << ii << " ";
            datasize *= ii;
        }
        desc_file << std::endl;

        std::ofstream oufile(input_name + std::to_string(count) + ".bin",
                             std::ios::binary);
        if (oufile) {
            oufile.write(reinterpret_cast<char *>(data),
                         datasize * sizeof(float));
            oufile.close();
        }
    }

    void init_kv_caches() {
        using namespace nncase;
        using namespace nncase::llm;
        using namespace nncase::runtime;

        const auto num_blocks = 1024;
        const auto max_model_len = 4096;

        paged_attention_scheduler_ = paged_attention_scheduler(
            std::in_place, paged_attention_config_, num_blocks, max_model_len,
            std::vector<int>{1});
    }

    nncase::tensor onForward(nncase::value_t input_ids) {
        using namespace nncase;
        using namespace nncase::llm;
        using namespace nncase::runtime;

        if (0) {
            fs::path dir_path = "calib";
            try {
                fs::create_directory(dir_path);
                std::cout << "Directory created successfully: " << dir_path
                          << std::endl;
            } catch (const fs::filesystem_error &e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            std::ofstream outputFile("calib/input_desc" +
                                     std::to_string(count) + ".txt");
            dump_input(outputFile, input_ids, "calib/input_ids_float", "fp32",
                       count);
            count += 1;
        }

        auto input_ids_tensor =
            input_ids.as<nncase::tensor>().unwrap_or_throw();
        auto input_ids_shape = input_ids_tensor->shape();
        int64_t query_lens = input_ids_shape[0];

        auto kv_cache = paged_attention_scheduler_->schedule(
            std::vector<long>{0}, std::vector<long>{query_lens});

        auto ref_type = nncase::reference_type_t(
            std::in_place, datatype_t::paged_attention_kv_cache);

        auto object_ptrs = new object_node *[1];
        object_ptrs[0] = kv_cache.detach();

        auto bytes_span =
            as_span<std::byte>(std::span<object_node *>(object_ptrs, 1));
        hrt::data_deleter_t deleter = [](std::byte *ptr) {
            auto object_ptrs = reinterpret_cast<object_node **>(ptr);
            nncase_object_release(object_ptrs[0]);
            delete[] object_ptrs;
        };

        auto runtime_tensor =
            hrt::create(ref_type, {}, bytes_span, deleter).unwrap_or_throw();

        std::array<nncase::value_t, 2> inputs{input_ids, runtime_tensor.impl()};
        return entry_function_->invoke(inputs)
            .unwrap_or_throw()
            .as<nncase::tensor>()
            .unwrap_or_throw();
    }

  private:
    nncase::runtime::interpreter interpreter_;
    nncase::runtime::runtime_function *entry_function_;
    nncase::llm::paged_attention_config paged_attention_config_;
    nncase::llm::paged_attention_scheduler paged_attention_scheduler_;
};

template <typename T>
static nncase::tensor _Input(const std::vector<int> &shape,
                             std::shared_ptr<RuntimeManager> rtmgr) {
    nncase::dims_t shape_int64(shape.begin(), shape.end());
    return nncase::runtime::hrt::create(
               std::is_same_v<T, float> ? nncase::dt_float32 : nncase::dt_int64,
               shape_int64, nncase::runtime::host_runtime_tensor::pool_shared)
        .unwrap_or_throw()
        .impl();
}

} // namespace Ort

#endif /* ORTWRAPPER_hpp */