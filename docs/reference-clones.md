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

## 与本轮对应的上游提交

拉过 `origin/main`（`66da8c4f..25054241`）之后，CPU 侧有几条和本仓库这一轮**撞在同一条路上**的提交。
列在这里是因为它们的做法验证了我们的取舍，而不是因为抄了它们：

| 上游提交 | 内容 | 与本仓库的关系 |
|---|---|---|
| `ff06794b` | `cpu_bmi2_shift_aot`：`RORX`/`SARX`/`SHLX`/`SHRX` 单独一个升举器 | 我们这轮加了同样的四条，走的是 `lift_one` 的通用路径而不是独立升举器——计数在第三操作数上、按目标宽度掩码，其余与 `shl` 无异，没有单独开文件的理由 |
| `9f13c507` | `cpu_sse_fp_aot`：legacy SSE2 的 `CMPSD`/`MOVLPD`/`MOVHPD`/`MOVNTPD` | 我们把 legacy 的 `movss xmm,xmm`（复制全部 128 位）与 `movss xmm,m32`（清高位）当成**两种语义**处理，宁可变窄也不猜 |
| `3d618698` | 修 SSE2 `MINSD`/`MAXSD` 在 DAZ（denormals-are-zero）下的行为、`XSAVE` 只存 AVX 的 MXCSR、MMX 读状态 | 我们的 `max`/`min` 写成 `(a > b ? a : b)` 而不是 `fmaxf`/`fminf`：SSE 的 `MAXSS` 在任一侧为 NaN 时返回**第二个**操作数，C 的那两个函数不是。DAZ 上我们的比较与硬件用的是同一条 `ucomiss`，所以随 MXCSR 一起走 |
| `c3221992` / `b291e792` | VEX.256 `VPACKSSWB` 系列、VEX.128 `VPADDB`/`VPCMPEQB`/`VPCMPGTB` | 我们还没做。它们在拒绝面里分别是 `vblendps`/`vpcmpeqb`/`vcmpeqps` 那一族（2,232 个）——需要一个能表达 128 位比较的中间表示，是下一个里程碑 |

一条**没有**采纳的差异：上游按指令族拆成多个升举器文件（`cpu_leaf_aot`、`cpu_avx_int_aot`、
`cpu_sse_fp_aot`、`cpu_bmi2_shift_aot`……）。本仓库只有一个 `lift_one`，因为这里的目标是先把
"翻译 → 编译 → 与硬件比"这个环跑通，而不是先长成一张指令表。指令数一旦过千，这个决定要重做。

## 一个还没建模的状态：MXCSR

整个 `.pdata` 里只有一对 `vldmxcsr`/`vstmxcsr`，所在函数我们不升举（正确拒绝），所以它留在了桥后面。
后果是：**舍入模式与 DAZ/FTZ 都没有作为状态进差分**。函数自己存了再恢复就没事；如果哪一处
`ldmxcsr` 改了模式却不恢复，被它影响的浮点结果就会落在"两侧进程各自的 MXCSR"上——和 TEB 一样，
属于量具的边界。要收掉它，得让 `vldmxcsr`/`vstmxcsr` 进模型并把 MXCSR 记进用例记录。

## 本轮又对上的三条上游提交（`25054241..1a0e506d` 区间）

拉了一次 `origin/main`，`25054241` 之后的 30 多条提交**全部是 GPU/着色器侧**
（mesh shader、SDWA、split sampler、depth-only pipeline……），x86 升举器没有新动作。
CPU 侧最后一条仍是 `780f3c0a`（x87 `F2XM1`/`FYL2X`/`FYL2XP1`，1,239 行升举器 + 1,031 行逐位校验），
本仓库的 x87 依然完全空白——这是接下来唯一"整族没做"的指令面。

| 上游提交 | 内容 | 与本仓库这一轮的关系 |
|---|---|---|
| `6a127266` | `cpu_bmi1_aot`：`BLSI`/`BLSMSK`/`BLSR` | 这一轮实现了同样的三条。有意思的是它们的**零输入语义**：三个在源为 0 时都给出定义好的结果，所以不能拒绝，也不能靠 `bsf` 的"未定义"含糊过去 |
| `b291e792` | VEX.128 `VPADDB`/`VPSUBB`/`VPCMPEQB`/`VPCMPGTB` | 这一轮把 `pcmpeq*`/`pcmpgt*` 做成了一张表（通道宽度 + C 关系），`vpsubb` 也进了 `VEC_INT_ARITH` | 
| `4ea0c770` | `cpu_avx_test_aot`：`VPTEST`/`VTESTPS`/`VTESTPD` | 这一轮做了，走 `branch.py` 的 `pending`（`kind == "vtest"`）。**没有照抄它注释里的归属**：SDM 是 `ZF = ((SRC1 AND SRC2) == 0)`、`CF = ((SRC2 AND NOT SRC1) == 0)`，而 VTESTPS/VTESTPD 只比**符号位**。按"CF 是 AND"去接，`jz` 之后整条路是反的，而且在 `vtestps x, x` 上看不出来 |

`4ea0c770` 正好说明这个仓库现在的分层在哪里卡住：**凡是写 EFLAGS 的 SIMD 指令，都不属于"向量段"**。
`vec_line` 的契约是"不碰标志"，所以 `vtestps`/`vptest`/`vucomiss` 都得在 `branch.py` 里有一条分支；
这一轮 `vtestps`/`vtestpd` 补上了（`kind == "vtest"`），并且在补的过程中被差分纠正了两次语义。

一条**仍然成立**的差异：上游把每一族做成独立文件 + 独立期望文件（`tests/cpu/*_check.py` 逐位比对），
本仓库是一条 `lift_one` + 一张 `vec_line`。这一轮 `vec_line` 从 120 行长到 400 行，
表驱动的部分（`VEC_BLEND`/`VEC_UNPACK`/`VEC_SHIFT`/`VEC_CMP_INT`/`VEC_HADD`）开始像一张指令表，
而"表之外还有分支"的部分（`movlps` 的 VEX 两操作数形式清零、`haddpd` 的排布、`psrad` 的饱和）
恰恰是最容易错的地方。拆不拆已经不是风格问题，而是"错的时候能不能只读一个文件就知道"。

## 又拉了一次上游：`1a0e506d..e4e2b46d`（160 条提交）

CPU 侧（`tools/cpu_*_aot.py` 与它们的期望文件）在这个区间里只有五条，其余全是
GPU/着色器、PE 解析、启动诊断与 CLI 加固：

| 上游提交 | 内容 | 与本仓库这一轮的关系 |
|---|---|---|
| `5f51daaa` | `op_sqrtsd_q` 不再用"C 符号判断 + `std::sqrt`"，改成用 `_mm_sqrt_sd` **在处理器上执行**：DAZ 下一个负的 denormal 被读成 -0、`SQRTSD` 返回 -0，而 C 的写法给的是 `0xFFF8000000000000`（实数不定值） | **同一处、我们这边还是旧的写法**。`VEC_SQRT` 走 `sqrtf`/`sqrt`，在默认 MXCSR、没有 NaN 载荷与 denormal 的输入上与硬件一致（片段差分 42 条里 `vsqrtps`/`vsqrtss`/`vsqrtsd` 都过），但 DAZ/FTZ 与 NaN quieting **没有进模型**——和 MXCSR 那条边界是同一件事。上游的解法很干脆：不建模，把这条指令交给处理器。这一条记进来是因为它说明"正确舍入"不等于"等价" |
| `141cc972` | `cpu_sha256_aot`：`SHA256RNDS2`/`SHA256MSG1`/`SHA256MSG2` | 映像里没有（`cover.py` 的拒绝面里一次都没出现），不需要做 |
| `d2cc0a79` | 在 opt-in 的宿主 CPU profile 下升举 `SGDT`/`SIDT`/`SLDT`/`SMSW`/`STR` | 我们一律拒绝：它们读的是**宿主**的 GDT/IDT，主机上跑的时候答案本来就不同 |
| `394f24f8` / `267b4be9` | CPU 模糊测试补上 x87 区域与位运算生成器 | 我们的对应物是 `lift.py --limit` 的发射器↔解释器自检 + `diff.py` 的两种粒度；模糊测试只覆盖"我们和我们对齐"，不覆盖"我们和硬件对齐" |
| `780f3c0a` | `cpu_x87_exp_log_aot`：`F2XM1`/`FYL2X`/`FYL2XP1` | 本仓库的 x87 **仍然完全空白**。这三个函数在 80 位扩展精度里算 `2^x-1` 与 `y·log2(x)`，用 `double` 复现会在第 64 位之后开始偏——整族没做是当前最大的一个"整块空白" |

顺带一条方法论上的对照：上游这一轮连开三条提交（`#2545`/`#2506`/`#1784`）都在**同一件事**上——
让每一条拒绝都有名字、有期望、有测试。本仓库的对应物是 `cover.py` 的拒绝面直方图与
`diff.py --snippets`：前者保证"拒绝的原因有名字"，后者保证"接受的指令真被执行过"。
