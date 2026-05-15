#include "rtp_llm/cpp/engine_base/grammar/GrammarLogitsProcessor.h"

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                               py::module_                        triton_bitmask_ops,
                                               GenerateStreamPtr                  stream):
    matcher_(std::move(matcher)), triton_bitmask_ops_(std::move(triton_bitmask_ops)), stream_(stream) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() {
    if (!triton_bitmask_ops_) {
        return;
    }
    if (Py_IsInitialized()) {
        py::gil_scoped_acquire acquire;
        triton_bitmask_ops_ = py::module_();
    } else {
        (void)triton_bitmask_ops_.release();
    }
}

DLTensor GrammarLogitsProcessor::makeBitmaskView(int32_t* data, int32_t batch_size, int32_t words) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    static thread_local int64_t shape[2];
    shape[0]       = batch_size;
    shape[1]       = words;
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

void GrammarLogitsProcessor::applyBitmaskToLogits(const py::module_& triton_bitmask_ops,
                                                   at::Tensor          logits,
                                                   const at::Tensor&   bitmask_cpu,
                                                   int32_t             vocab_size) {
    RTP_LLM_PROFILE_SCOPE("grammar.bitmask_to_gpu+apply");
    auto bitmask_gpu    = bitmask_cpu.to(logits.device(), /*non_blocking=*/true);
    auto target_logits  = logits.size(1) > vocab_size ? logits.slice(/*dim=*/1, 0, vocab_size) : logits;
    py::gil_scoped_acquire acquire;
    triton_bitmask_ops.attr("apply_token_bitmask_inplace_triton")(
        convertTensorToObject(target_logits), convertTensorToObject(bitmask_gpu));
}

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_ || matcher_->isTerminated() || matcher_->finished() || matcher_->isPassthroughForMask()) {
        return;
    }

    if (!triton_bitmask_ops_) {
        if (!Py_IsInitialized()) {
            return;
        }
        if (auto stream = stream_.lock()) {
            stream->reportError(ErrorCode::EXECUTION_EXCEPTION,
                                "grammar bitmask kernel unavailable: triton import failed "
                                "at engine init (see prior WARNING). Verify triton install.");
        }
        return;
    }

    const int batch_size = static_cast<int>(finish_idx - start_idx);
    const int vocab_size = matcher_->vocabSize();
    const int words      = (vocab_size + 31) / 32;

    auto     bitmask = at::full({batch_size, words}, /*fill_value=*/-1, at::dtype(at::kInt));
    DLTensor dl      = makeBitmaskView(bitmask.data_ptr<int32_t>(), batch_size, words);

    {
        RTP_LLM_PROFILE_SCOPE("grammar.fillBitmask");
        for (int i = 0; i < batch_size; ++i) {
            matcher_->fillBitmask(&dl, i);
        }
    }

    auto logits_slice = inputs.logits.narrow(0, start_idx, batch_size);
    applyBitmaskToLogits(triton_bitmask_ops_, logits_slice, bitmask, vocab_size);

    matcher_->mutableStats().mask_apply_count++;
}

void GrammarLogitsProcessor::acceptTokens(const std::vector<int32_t>& tokens) {
    if (!matcher_ || matcher_->isTerminated() || matcher_->finished()) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    for (int32_t tok : tokens) {
        if (!matcher_->acceptToken(tok)) {
            RTP_LLM_LOG_WARNING("[grammar] parser rejected token %d", tok);
            if (auto stream = stream_.lock()) {
                stream->reportError(ErrorCode::INVALID_PARAMS,
                                    "grammar accept_token error: parser rejected token "
                                        + std::to_string(tok));
            }
            return;
        }
    }

    checkTermination();
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_ || matcher_->isTerminated() || matcher_->finished()) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    const int batch_size = static_cast<int>(new_tokens.size(0));

    std::vector<int32_t> tokens;
    tokens.reserve(batch_size * num_new_tokens);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_new_tokens; ++j) {
            tokens.push_back(new_tokens.data_ptr<int32_t>()[i * new_tokens.size(1) + j]);
        }
    }
    acceptTokens(tokens);
}

void GrammarLogitsProcessor::checkTermination() {
    if (matcher_->isTerminated()) {
        matcher_->markFinished();
        if (auto stream = stream_.lock()) {
            if (stream->isActive()) {
                stream->reportEvent(StreamEvents::GenerateDone);
            }
        }
    } else {
        if (auto stream = stream_.lock()) {
            if (!stream->isActive()) {
                matcher_->markFinished();
            }
        }
    }
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

}  // namespace rtp_llm
