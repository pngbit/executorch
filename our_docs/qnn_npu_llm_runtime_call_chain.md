# ExecuTorch + Qualcomm QNN NPU 运行 LLM 的完整调用链：CPU 与 NPU 分工

本文基于当前仓库 `pngbit/executorch` 中的以下代码整理：

- `extension/llm/runner/text_llm_runner.cpp`
- `extension/llm/runner/text_prefiller.cpp`
- `extension/llm/runner/text_token_generator.h`
- `extension/llm/runner/text_decoder_runner.cpp/.h`
- `extension/llm/runner/io_manager/io_manager.h`
- `extension/module/module.cpp`
- `runtime/executor/method.cpp`
- `backends/qualcomm/partition/qnn_part