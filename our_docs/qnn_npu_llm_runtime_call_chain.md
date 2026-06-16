# ExecuTorch + Qualcomm QNN NPU 运行 LLM 的完整调用链：CPU 与 NPU 分工

本文整理当前 `pngbit/executorch` 中 Android/端侧 LLM 通过 ExecuTorch + Qualcomm QNN 后端运行时，从输入 prompt 到输出文本的完整链路，并标注哪些阶段在 CPU 上执行，哪些阶段可能进入 QNN HTP/NPU。

> 结论先行：`generate()`、tokenizer、sampler、decode 外层循环、ExecuTorch instruction dispatch、I/O 准备、QNN backend 初始化与内存绑定都在 CPU 侧。真正进入高通 NPU/HTP 的只有 `.pte` 中被 AOT QNN partition/lowering 成 `DelegateCall` 的神经网络子图。若 `.pte` 中仍存在 `KernelCall`，这些就是 CPU fallback/portable kernel 路径。

## 1. 总体链路

```text
Android App / JNI / native caller
  -> TextLLMRunner::generate(prompt, config, callback)
    -> load(): Program/Method/QNN backend/tokenizer/IOManager 初始化
    -> tokenizer.encode(prompt)                         [CPU]
    -> TextPrefiller::prefill(prompt_tokens, pos_)       [CPU 控制]
       -> TextDecoderRunner::step(tokens, start_pos)     [CPU 控制]
          -> IOManager::prepare_decode(...)              [CPU]
          -> Module::execute(method_name, inputs)        [CPU 进入 executor]
             -> Method::execute() 顺序解释 chains/instructions [CPU]
                -> KernelCall                           [CPU fallback kernel]
                -> DelegateCall                         [CPU 调 QNN backend]
                   -> QnnExecuTorchBackend::execute(...) [CPU 绑定输入输出]
                      -> QnnManager::Execute(...)
                         -> qnn_graph_ptr_->GraphExecute(...) [QNN HTP/NPU]
          -> IOManager::update_decode(...)               [CPU]
          -> return logits tensor
       -> logits_to_token / sampler                      [CPU]
    -> tokenizer.decode(first token)                     [CPU]
    -> TextTokenGenerator::generate(...) decode loop      [CPU 控制]
       -> 每生成一个 token 调一次 TextDecoderRunner::step [NPU 子图 + CPU 控制]
       -> logit processors / sampler / EOS 判断 / decode  [CPU]
       -> token_callback(piece)                          [CPU]
```

## 2. 输入、加载、tokenize 阶段

### CPU 执行

`TextLLMRunner::generate()` 是文本生成主入口。它先判断是否加载，未加载则调用 `load()`；`load()` 依次加载 `text_prefiller_`、`io_manager_`、`text_token_generator_`。随后 `generate()` 读取 `max_seq_len`、`max_context_len`，调用 tokenizer 把字符串 prompt 编成 token 向量。

对应代码：

- `TextLLMRunner::load()`：`text_prefiller_->load()`、`io_manager_->load()`、`text_token_generator_->load()`。
- `TextLLMRunner::generate()`：`tokenizer_->encode(prompt, ...)`、prompt 长度和 context 长度检查、进入 prefill。

这些逻辑都是普通 C++ 逻辑，运行在 CPU 上；NPU 不处理字符串、tokenizer、配置检查或回调。

## 3. Prefill 阶段

### CPU 控制

`TextPrefiller::prefill()` 会根据 prompt token 数量和 `max_seq_len_` 决定是否切 chunk：

```text
prompt_tokens.size() <= max_seq_len_  -> prefill_chunk(prompt_tokens)
prompt_tokens.size() >  max_seq_len_  -> 多个 chunk 循环 prefill_chunk(...)
```

这说明：

- 单个固定 shape/chunk 的 QNN graph 计算量可以接近固定；
- 但整个 prefill 的调用次数会随 prompt 长度变化；
- 短 prompt 如果被 padding 到固定长度，会有无效计算；
- 长 prompt 会被拆成多个 chunk，整体计算量不是恒定的。

### 可能进入 NPU 的部分

`prefill_chunk()` 最终调用 `text_decoder_runner_->step(tokens, start_pos)`。`step()` 内的 `Module::execute()` 会运行 `.pte` 方法；如果该方法的 transformer/prefill 子图被 QNN delegate，则这些神经网络算子在 `DelegateCall -> QNN -> GraphExecute` 内执行，进入 HTP/NPU。否则，未 delegate 的部分会以 `KernelCall` 在 CPU 上执行。

## 4. Decode 循环阶段

### CPU 控制

`TextTokenGenerator::generate()` 是 decode 主循环。KV cache 模式下，代码把 token shape 固定为 `{1, 1}`，注释写明 “kv cache is locked to static size right now”。循环中每次：

1. 调 `text_decoder_runner_->step(tokens_managed, pos)`；
2. 对 logits 应用 logit processors；
3. 调 `logits_to_token()` 采样下一个 token；
4. 更新 `token_data[0]`；
5. tokenizer decode 成字符串；
6. 调 callback；
7. 检查 stop/EOS。

上述循环、采样、EOS 判断、字符串 decode 和 callback 都在 CPU。

### 可能进入 NPU 的部分

每次 `step()` 内的 `Module::execute()` 会再次进入 ExecuTorch executor。如果 decode graph 被 QNN delegate，则每一步 decode 的神经网络前向在 NPU/HTP 执行；但外层循环本身仍由 CPU 调度。

因此：

```text
生成 1 个 token  -> 约 1 次 decode graph execute
生成 N 个 token  -> 约 N 次 decode graph execute
```

在同一个固定 shape decode graph 内，每步计算形状可以固定，可能存在 padding/mask 带来的无效计算；但输出越长，总 decode 次数仍然线性增加。

## 5. TextDecoderRunner::step()

`TextDecoderRunner::step()` 是 CPU 与模型执行之间的关键桥接层。

KV cache 模式：

```text
method_meta = module_->method_meta(method_name_)
use_kv_cache = method_meta.num_inputs() > 1
populate_start_pos_or_cache_position(...)
io_manager_->prepare_decode(tokens, start_pos_tensor, method_name_)
module_->execute(method_name_, inputs)
io_manager_->update_decode(outputs, method_name_)
return logits
```

这些准备和更新逻辑都在 CPU。是否进入 NPU，取决于 `module_->execute()` 内部执行的是 `DelegateCall` 还是 `KernelCall`。

## 6. Module::execute() 与 Method::execute()

`Module::execute()` 做三件事：

1. `load_method(method_name)`；
2. 对每个输入调用 `method->set_input(...)`；
3. 调 `method->execute()`；
4. 调 `method->get_outputs(...)` 取输出。

这些都是 CPU 侧 runtime/executor 逻辑。

`Method::execute()` 会顺序遍历 execution plan 的 chains 和 instructions。关键分支：

```text
KernelCall   -> 调用已注册 CPU kernel / portable/custom kernel
DelegateCall -> 调用 delegates_[delegate_idx].Execute(...)
MoveCall     -> CPU 侧 EValue 移动
FreeCall     -> CPU 侧释放/重置 tensor data ptr
JumpFalse    -> CPU 侧控制流判断
```

所以判断是否还有 CPU fallback 的最重要依据是 `.pte` 的 instruction：

- 如果模型主图仍有 `KernelCall`，就需要 CPU kernel，属于 CPU 执行；
- 如果模型主图被完整 lower 成一个或多个 QNN `DelegateCall`，神经网络计算才会走 QNN backend；
- 即使 100% delegate，executor 仍在 CPU 上负责 instruction dispatch 和 delegate 调用。

## 7. QNN backend 执行阶段

当 executor 遇到 `DelegateCall` 后，会调用 QNN backend。

`QnnExecuTorchBackend::init()`：

- 从 `.pte` delegate processed data 中反序列化 QNN context binary 或 DLC；
- 解析 compile spec；
- 创建 `QnnManager`；
- 初始化 QNN backend/context；
- online prepare 时编译 DLC，否则为 context binary 里的 graph 分配 tensor。

这些初始化逻辑在 CPU 上执行。

`QnnExecuTorchBackend::execute()`：

- 根据 method name 获取 QNN graph inputs/outputs；
- 将 ExecuTorch tensor data pointer 注册/填入 QNN tensor；
- 处理输出 tensor buffer；
- 调 `qnn_manager->Execute(method_name, input_tensor_structs, output_tensor_structs, ...)`。

这部分仍然是 CPU 侧 QNN runtime glue。真正的图执行发生在 `QnnManager::Execute()` 内：

```cpp
backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
    graph_name, input_tensor_structs, output_tensor_structs);
```

这一步才是 QNN graph 的执行入口；使用 HTP backend 时，QNN graph 内的已支持算子由高通 HTP/NPU 执行。

## 8. QNN partition 决定哪些算子能进 NPU

AOT 编译时，`QnnPartitioner` 负责识别哪些 FX node/subgraph 可以 lower 到 QNN backend。

`QnnOperatorSupport::is_node_supported()` 的判断逻辑包括：

- 非 `call_function` 或 `not_supported_operator`：不支持；
- `to_be_implemented_operator`：当前未实现；
- node meta 标记 `QCOM_FALLBACK_NODE`：强制 fallback；
- allow list/custom op/bypass node：可强制通过；
- 被 skip id/op 命中：跳过；
- 否则构造 QNN op wrapper，并调用 `qnn_manager.IsNodeSupportedByBackend(...)` 让 QNN backend 验证 op config。

因此，运行时 CPU/NPU 分工在 AOT 阶段就基本决定了：能 partition 成 QNN delegate 的成为 `DelegateCall`；不能 partition 的保留为 `KernelCall`，运行时走 CPU。

## 9. 当前链路中哪些必须保留 CPU

即使追求“模型主干 NPU-only”，以下部分仍然必须保留在 CPU/Host 侧：

- Android UI / JNI / native 调用入口；
- prompt 字符串处理；
- tokenizer encode/decode；
- generation config、长度检查、EOS/stop 判断；
- prefill chunking 和 decode 外层循环；
- sampler / temperature / logit processors；
- ExecuTorch runtime、Program/Method 加载、instruction dispatch；
- QNN backend 初始化、context 管理、输入输出 tensor 绑定；
- token callback、stdout/UI 输出。

这些不是“CPU fallback 算子”，而是主机控制逻辑，不能因为模型上 NPU 就删除。

## 10. 可以通过 NPU-only 目标删除/裁剪的 CPU 部分

如果你能证明某个 `.pte` 的神经网络计算图没有任何 `KernelCall` fallback，那么可以考虑裁剪：

- 未使用的 portable CPU operator kernels；
- 未使用的 custom CPU kernels；
- 未使用的 fallback op registration；
- 与该模型无关的 kernel libraries。

但不能删除：

- ExecuTorch executor 本体；
- QNN delegate runtime；
- tokenizer/sampler/loop；
- QNN SDK/HTP backend 所需 host 侧库；
- Android native glue。

## 11. 如何验证是否还有 CPU fallback

建议检查三层：

1. AOT partition log：查看哪些 node 被 `QnnPartitioner` 标成 QNN delegate，哪些 node 打印 `False`、`Skipped`、`Forced Fallback`。
2. `.pte` execution plan：确认模型主图是否仍有 `KernelCall` 指向 aten/portable op。
3. 运行时 profiler/ETDump/QNN profile：确认 token 生成期间是否仍有 `OPERATOR_CALL` 时间，而不是只有 `DELEGATE_CALL` + QNN graph 时间。

如果目标是“LLM 神经网络计算不降级 CPU”，验收标准应写成：

```text
prefill method: 0 个模型计算 KernelCall，全部为 QNN DelegateCall
每步 decode method: 0 个模型计算 KernelCall，全部为 QNN DelegateCall
CPU 仅负责 tokenizer / sampler / loop / executor dispatch / QNN host glue
```

## 12. 最终分类表

| 阶段 | 代码位置 | CPU / NPU | 说明 |
|---|---|---|---|
| App 输入 prompt | Android/JNI/native caller | CPU | UI、字符串、调用 native runner |
| `TextLLMRunner::generate` | `extension/llm/runner/text_llm_runner.cpp` | CPU | 主控流程、加载、统计、长度检查 |
| tokenizer encode | tokenizer object | CPU | 字符串到 token |
| prefill chunking | `TextPrefiller::prefill` | CPU | 切 chunk、更新 `start_pos` |
| prefill graph execute | `TextDecoderRunner::step -> Module::execute` | NPU 或 CPU fallback | 取决于 `.pte` 中是 QNN `DelegateCall` 还是 `KernelCall` |
| decode loop | `TextTokenGenerator::generate` | CPU | 每 token 一轮循环 |
| decode graph execute | `TextDecoderRunner::step -> Module::execute` | NPU 或 CPU fallback | QNN delegated subgraph 进 NPU |
| sampler/logits_to_token | sampler/logit processors | CPU | 采样、温度、logit 处理 |
| tokenizer decode | tokenizer object | CPU | token 到字符串 |
| callback/output | `token_callback` | CPU | stdout/UI 输出 |
| executor dispatch | `runtime/executor/method.cpp` | CPU | 顺序解释 instructions |
| `KernelCall` | `Method::execute_instruction` | CPU | CPU fallback kernel |
| `DelegateCall` host glue | `QnnExecuTorchBackend::execute` | CPU | 输入输出绑定、调用 QNN |
| QNN graph | `QnnManager::Execute -> GraphExecute` | QNN HTP/NPU | 真正 NPU 执行入口 |

## 13. 对你的减包体积目标的建议

如果目标是减小 Android 包体积，不应该试图删除所有 CPU 代码，而应该区分：

```text
必须保留：CPU 控制层 + ExecuTorch executor + QNN host runtime + tokenizer/sampler
可以裁剪：未使用的 CPU fallback operator kernels
```

落地路径：

1. 固定支持的 LLM 架构、shape、quantization、KV cache layout；
2. AOT 阶段用 QNN partition log 找所有 fallback；
3. 对 fallback 算子做 graph rewrite、QNN NodeVisitor 或 QNN custom op package；
4. 确认 prefill/decode `.pte` 中没有模型计算 `KernelCall`；
5. 做 selective build，只保留 tokenizer/sampler/runner 必需 CPU 代码和 QNN delegate；
6. 用 ETDump/QNN profile 对比 `OPERATOR_CALL` 与 `DELEGATE_CALL`，确认 CPU fallback 已消除。
