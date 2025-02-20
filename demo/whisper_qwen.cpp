#include "llm.hpp"
#include "whisper.h"
#include <dirent.h>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>

void benchmark(Llm *llm, std::string prompt)
{
    std::vector<std::string> prompts{prompt};

    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    for (int i = 0; i < prompts.size(); i++)
    {
        auto result = llm->response(prompts[i]);
        prompt_len += llm->prompt_len_;
        decode_len += llm->gen_seq_len_;
        prefill_time += llm->prefill_us_;
        decode_time += llm->decode_us_;
    }
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", prompt_len);
    printf("decode tokens num  = %d\n", decode_len);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt | dataset_path number>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;

    std::string whisper_model = argv[2];
    std::string wav_path = argv[3];
    auto prompt = whisper_get_sentence(whisper_model, wav_path);

    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load();

    if (argc == 4)
    {
        
        benchmark(llm.get(), prompt);
    }
   

    return 0;
}
