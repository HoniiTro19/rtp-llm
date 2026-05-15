#pragma once

#include <memory>

#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace py = pybind11;

namespace rtp_llm {

class RtpGrammarMatcher;
class GenerateStream;
using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

class GrammarLogitsProcessor: public BaseLogitsProcessor {
public:
    GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                           py::module_                        triton_bitmask_ops,
                           GenerateStreamPtr                  stream);

    ~GrammarLogitsProcessor() override;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;

    RtpGrammarMatcher* grammarMatcher() const override { return matcher_.get(); }

private:
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    py::module_                        triton_bitmask_ops_;
    std::weak_ptr<GenerateStream>      stream_;
};

}  // namespace rtp_llm
