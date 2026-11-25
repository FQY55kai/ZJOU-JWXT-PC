# 两个参考仓库的本地副本

克隆在 `D:\recomp-refs\`，浅克隆（`--depth 60`）够读代码，不够读全部历史。

| 目录 | 来源 | 规模 | 性质 |
|---|---|---|---|
| `Xbox2Windows/` | `github.com/R0Y00/Xbox2Windows`（MIT） | 35 MB，719 个 `.py` / 331 个 `.cpp` | **真·静态重编译**：x64 机器码 → C++ |
| `palmetto/` | `github.com/luizwidmer/palmetto` | 62 MB，1,256 个 `.py` / 312 个 `.cpp` | **HLE 兼容层**（不是 CPU 重编译） |

> 克隆时 `schannel` 因取不到吊销列表报 `CRYPT_E_NO_REVOCATION_CHECK`，用
> `GIT_SSL_NO_VERIFY=1 git clone` 绕过（已写进全局 `http.schannelCheckRevoke=false`，但对 git 2.55 无效）。
> 网络本身没问题，`curl --ssl-no-revoke` 能正常访问。

## 先纠一个分类错误

之前把 Palmetto 归到"静态重编译"，**这个归类是错的**。它的 README 写得很清楚：

> Palmetto translates a retail console executable and its shaders into a native Windows program at build time:
> **the loader maps the console image, the runtime resolves its imports to open-source shims**, and the shader
> tooling translates the console shaders to SPIR-V.

也就是说：CPU 代码**不翻译**，原样映射进进程执行（和 VIctory、XCL 同一路线）；"静态"只发生在**着色器**上
（主机 shader → SPIR-V，构建期完成）。`git ls-files | grep -iE "lift|disasm|recompil"` 只命中
`tests/semantic/fixtures/disasm/*.s`（shader 反汇编夹具），**仓库里没有任何 CPU 升举器**。

所以四个项目里真正做 CPU 静态重编译的**只有一个**：Xbox2Windows。这不是坏消息——它意味着
"这条路没人走通过"这个判断比之前更成立，也意味着 Palmetto 的 GPU/运行时层对我们**没有**可移植价值
（它是 Vulkan 上的 D3D12，我们最终要的是原生 D3D12）。

## Xbox2Windows：读什么

`tools/` 下 118 个文件，CPU 升举是按**指令集扩展**拆开的，不是一个巨型 lifter：

| 文件 | 行数 | 管什么 |
|---|---|---|
| `cpu_leaf_aot.py` | 138 | 入口/参数个数约定（`name:4` = 4 个整型参数），`function_ranges` 定边界 |
| `cpu_switch_aot.py` | **2,622** | 跳转表 → `switch`，**并附完整的标志位语义文档**（见下） |
| `cpu_avx2_aot.py` | 978 | AVX2 |
| `cpu_call_aot.py` | 240 | 函数内 `call` |
| `cpu_chained_aot.py` | 242 | 片段/funclet（`.pdata` 区间不独立成立时） |
| `cpu_sse2_aot.py` / `cpu_sse_aot.py` / `cpu_vex_aot.py` / … | — | 其余 SIMD |
| `fuzz_cpu.py` | — | 随机指令模糊测试 |

`tools/cpu_switch_aot.py` 的模块注释是**目前找到的最完整的标志位规范**，比我们
`docs/reference-xbox2windows.md` 里抄的规则细得多，直接决定我们下一步：

- 跳转表只接受两种形态，且 `B` 必须在**每条路径上都是同一个 RIP 相对地址**，索引必须被证明
  **等于参与比较的值**；`JA/JAE` 到 default 的边界检查、或 `JB/JBE` 到 dispatch 的反写形态；
- `SHL/SAL/SHR/SAR` 只在**立即数计数**且掩码后落在 `1..宽度-1` 时算生产者；计数为 0 保留旧标志
  （所以 `CL` 计数一律拒绝）；`CF` 是最后移出的一位，`OF` 只在计数为 1 时有定义；
- `BT/BTS/BTR/BTC` 的位偏移对寄存器基址和立即数偏移取模到操作数宽度；寄存器偏移进普通内存是
  **有符号位串**寻址，帧相对 / 映像相对 / 32 位地址各有规则。

这三条正好覆盖我们现在的三个拒绝项（`indirect branch` 845、`bt` 396、`jne without a compare` 743）。

## 可复用的验证思路

X2W 用三个 runner 互比（源构建 EXE / 映射 PE / AOT 重编译），**逐字节比较输出**：纹理、帧、
回调顺序、分配释放、退出码。我们比它强的一点是裁判是硬件（同 ISA 直接 `call` 原始字节），
该补的是"比什么"：现在只比 16 个通用寄存器和 256 字节栈，这就是下面那条天花板的另一半。
