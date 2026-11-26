# 参照：Xbox2Windows（R0Y00，MIT）

X2W 是目前**唯一**明确自述"x64 机器码 → C++ 静态重编译"并持续活跃的项目（2.6k+ 提交）。
我们读了它的 `tools/`（108 个脚本）和 `src/`（运行时 shim），下面只记**对我们要做的决定有影响的部分**。

## 它的形状

| 部分 | 做法 |
|---|---|
| 取边界 | `.pdata` + `RtlAddFunctionTable` 注册（`tools/pe_map.cpp`）；与我们一致 |
| 升举单元 | **bounded leaf 函数**：参数只在 RCX/RDX/R8（:4 时加 R9），只用整数，栈参数与浮点参数不支持 |
| 标志位 | **只在"生产者 → 消费者"之间建模**，且该直线段**没有入边**；`Jcc`/`SETcc`/`CMOVcc` 是消费者；`mov`/`lea`/`not`/`bswap`/`cbw/cwde/cdqe/cdq/cqo` 不破坏标志，可以夹在中间 |
| 溢出/奇偶 | 用**生产者的操作数宽度**判，含窄算术（8/16 位） |
| 跳转表 | 严格模式匹配后转 `switch`：`mov E32,[B+I*4+table]; add E,B; jmp E` 或 `movsxd E,[B+I*4]; add E,B; jmp E`；`B` 必须在每条路径上都是同一个 RIP 相对地址，索引必须被证明等于参与比较的值 |
| 数据 vs 代码 | 只从函数里跟着分支/跳转表走，**不线性扫描**——MSVC 会把表放在代码后面 |
| 验证 | 三个 runner 对比：源构建 EXE / 映射 PE / AOT 重编译，输出**逐字节**比较（纹理、帧、回调顺序、分配释放、退出码） |
| 图形 | gfx1013 → DXIL，26,500 / 26,503 条 shader 记录找到编译路径 |

## 我们采纳的

1. **标志位按"生产者→消费者、无入边直线段"建模**——这正是我们 `branch.py` 缺的那一块：现在只认 `cmp/test` 紧跟 `jcc`，X2W 的规则允许中间夹 `mov/lea/not`，并且把 `setcc`/`cmovcc` 也算作消费者。这是 M2 的正确第一步，比"全局标志寄存器"便宜得多。
2. **跳转表转 `switch`**，用他们的模式；拒绝条件也照抄（`B` 必须同址、索引必须可证）。
3. **三个 runner 对比**：我们已经有"映射 PE = oracle"，再补一个"源构建"没有意义（我们没有源码），但**逐字节比较输出**这一点要学：不只比寄存器，还要比被写内存的哈希。
4. **参数个数由调用方给定**，不从机器码推导——避免"猜参数"这一类错误。

## 我们不同的

| | X2W | 我们 |
|---|---|---|
| 验证裁判 | 三个 runner 互比（都是自己的构建） | **原始机器码直接 `call`**（同 ISA），裁判是硬件 |
| 起点 | 先做图形与 shader，CPU 只做 bounded leaf + switch | 先把 CPU 侧覆盖率做实，图形留到 L5 |
| 目标 | "能启动、能出像素" | "最终超过 30 fps"（因此把渲染器替换当成 L5 的核心） |

## 立刻可做（照抄这两条）

- [ ] `flags.py`：直线段内的生产者→消费者标志位模型，覆盖 `Jcc`/`SETcc`/`CMOVcc`，含 8/16 位窄算术的 OF/PF
- [ ] `switch.py`：两种跳转表模式 → `switch`，`B` 同址与索引可证的检查照抄
- [ ] 差分比较加上"被写内存页的哈希"，不只比 16 个寄存器与 256 字节栈

## 读上游这一轮（`418f9245..1ec01d70`）：18 条提交，**CPU 侧 0 条**

先说结论：这 18 条里**没有一条**碰机器码升举。全部落在 `gfx2hlsl`（GFX ISA → HLSL）与
`startup`（描述符/资源状态）两块。这不是偶然——它说明上游自己的重心已经从"把 x64 翻成 C"
挪到了"把主机的 GPU 侧翻译掉"。我们这一轮也因此换了个方向：先数清楚 HLE 面（见 README
「HLE 面」与 `tools/hle.py`），而不是继续在 0.1% 的函数上磨。

| 组 | 提交 | 做的事 |
|---|---|---|
| GFX 指令覆盖（8 条） | `601eb621` `a766f287` `091c2a48` `8ece2649` `a768ebe0` `e6902a11` `1ec01d70` `651577a0` | `v_bfm_b32` 与缺失的 VOP2/VOP3；`v_sad_hi_u8`/`v_msad_u8`；compute VOP1 的 `bfrev`/`ffbh_i32`；`s_quadmask_b64`；`s_bfm_b64`；wave64 通道上的常量 lane SGPR 溢出；像素程序里带均匀分支/循环的 **DPP quad** 读；纯整数 VALU 之后**保持 wave-uniformity** |
| 描述符与资源（4 条） | `786970ac` `caa24ad7` `83331487` `b408241e` | 没有 unordered access 的资源其 UAV 记为"未创建"；描述符字复制进 `vcc` 当作 `s[104:107]` 的上半；接受 compute waterfall 之后读的 key-independent body scalar；placed resource 包装分配失败 |
| 诊断与测试（6 条） | `a5817910` `434aa90d` `7f2ed4ce` `6bf9bf1a` `fd033046` `2a9c252f` | `src_scc` 的 WARP 回读；给不支持的 implicit gather mip level **起个名字**（而不是报个数字）；一次表里两个复制的 raw vertex SRV；在 GPU 检查包里重放编译后的 compute 用例；并行负载下 `startup_run_ps1_check` 稳定；`startup_resource_address_index` 不受 MSVC Debug 迭代器检查影响 |

三条值得记的：

1. **"给错误起个名字"也算一条提交**（`434aa90d`）。我们没有这一步：`Refused` 的消息是给
   直方图用的，而 `why.py` 得从异常回溯里把真凶挖出来。命名一次，比事后猜一次便宜。
2. **wave-uniformity 是显式跟踪的属性**（`651577a0`），不是"看着像就行"。我们的对照物是
   标志位的 token——同一个思路，只是他们跟的是"这个 lane 上的值对所有 lane 相同"。
3. **测试稳定性是正经提交**（`fd033046`、`2a9c252f`）。我们的 `--rounds` 与 `SNIPPET_SEED`
   是同一个问题的另一半：不确定的裁判等于没有裁判。

## 读上游这一轮（`2a801192..418f9245`）：34 条提交，CPU 侧 3 条

| 提交 | 内容 | 能不能吸收 |
|---|---|---|
| `f14219f2` | `cpu: lift legacy SSE and VEX.128 AES-NI and PCLMULQDQ leaf functions to C++ (#2677)`——AES 轮函数族 + `PCLMULQDQ`（无进位乘法） | **吸收了一条**。这个映像里只出现 `vaesimc`（13 条，`aesimc` 的 VEX 形式，逆 MixColumns），落在约 10 个函数上；`aesenc`/`aesdec`/`aesenclast`/`aesdeclast` 一并接上（同一张表），`PCLMULQDQ` **这个映像里一次都没有**，不写。做法与 §5⑤ 一致：`_mm_aesimc_si128`/`_mm_aesenc_si128` 就是那些指令，AES 的 S 盒不该在 C 里重抄一遍 |
| `22013b20` | `cpu: add legacy SSE4.1 FP AOT lifter and test suite (#2691)`——legacy（无 VEX）形式的 `ROUNDPS/PD`、`BLENDPS/PD`、`INSERTPS`、`PMOVSX*`/`PMOVZX*`、`PACKUSDW`、`PBLENDW`… | **已覆盖**。本仓库的表全部按**去 VEX 前缀的 stem** 索引，legacy 与 VEX 走同一条发射路径——`roundps`/`blendps`/`insertps`/`pmovsx*` 的两操作数形式本来就在（`VEC_ROUND`/`VEC_BLEND`/`VEC_EXTEND`）。它的价值在于**提醒**：同一张表被两种编码共用时，"目的寄存器同时是源"这条只有 legacy 形式成立，VEX 三操作数形式的目的不是源（这条我们上一轮在 `vpsrldq` 上踩过，见 README §5⑧） |
| `ede9e4d3` | `cpu: lift 16/32/64-bit MOVBE load and store leaf functions to C++ (#2689)` | **已覆盖，但它订正了我们的一句注释**。`movbe` 的三种宽度本仓库都有（`_byteswap_ushort/ulong/uint64`），可注释里写着"只有 32/64 位有这条指令，16 位编码未定义"——**错的**：16 位带 `66` 前缀，SDM `MOVBE` 明确有 `r16/m16`，这个映像里 71 条。注释已改；代码本来就对，错的是说法 |

其余 31 条是 runtime / tooling / shader / CI（`62946ccf`、`547a8c67`、`a9cfecf0` 一带是给各 `*_aot` CLI 补原子落盘与参数校验，
`fd368513` 是 `pixel_aot` 的 binary16 辅助函数），没有指令语义。原子落盘本仓库早已在做（差分前写 `snippets.exe` 副本）。

两条对照：

- 上游把 AES 单独拆一个模块，本仓库是 `VEC_AES` 一张五条的表 + `vec_line` 的一个分支。判据仍是那句：
  一张表能表达的就不该是一个模块——五条里只有"一源还是两源"不同，而这一点用表里的第二个字段就写完了。
- 上游给 `PCLMULQDQ` 也配了实现，本仓库**不写**：这个映像里一次都没出现。这与 `PDEP`/`PEXT` 的处理一致——
  用不上的实现比"看起来完整"的表更贵（每一次覆盖扫描都要多算一遍）。

## 读上游上一轮（`fbe249f6..2a801192`）：57 条提交，CPU 侧 2 条

| 提交 | 内容 | 能不能吸收 |
|---|---|---|
| `41ad992e` | `cpu: add AVX VEX FP compare AOT lifter and test suite (#2671)`——`VCMPPS/PD/SS/SD` 的 128/256 位形式，32 个谓词，630 行升举器 + 1,309 行测试 | **已覆盖**。32 个谓词本仓库上一轮就做完了（`vcmpeqps`…`vcmptrue_usps`，片段差分逐条判过）。**但它的做法里有一条值得抄**：每个谓词都用 `volatile` 副本喂给基线 SSE 比较，注释写明是"防止 `/O2` 把 `+0/-0`、Inf、NaN、DAZ 或者 IE 信号折叠掉"。我们这边没有做这一步——因为比较走的是 `_mm_cmp*_ps` 而不是 C 的 `<`，本来就没有可折叠的表达式；这一条记在这里，作为"如果哪天改成裸比较，必须补 volatile"的备忘 |
| `404446e8` | `cpu: lift BMI1 ANDN, BEXTR, and TZCNT leaf functions to C++ (#2668)`——670 行升举器 + 803 行测试 | **已覆盖，而且它的拒绝面注释证实了我们踩的坑**：文档里写着 "ANDN leaves PF and AF undefined"、"BEXTR leaves SF, PF, and AF undefined"、"TZCNT leaves SF, OF, PF, and AF undefined"——本仓库的 `alu_shape` 对这三个名字的处理正好是同一个结论（`bextr` 的 SF 记作 `None`、`bls*` 的 CF 拒绝、`tzcnt` 只承认 z/c）。真正被这一条**提醒**到的不是指令，而是**标志的"未定义"要当拒绝，不能当零**——顺着这条把 `mul`/`imul` 的 CF/OF 补了（见下） |

其余 55 条是 shader / startup / d3d12 / CI / 测试，没有指令语义可言。

两条对照：

- 上游每个升举器都自带一套 `.asm` + `.def` + `_check.py` 的"指令级"套件（这一轮两条各 1,300 行与 800 行），
  本仓库对应的是 `diff.py --snippets`——同样的形态（**一条指令一个函数，跟硬件比**），
  差别在于他们比自己两套实现的输出，我们比硬件。这一点没变。
- 上游这一轮**没有**动 `div`/`cmpxchg16b`/x87，与本仓库剩下的拒绝面完全重合：
  三个尾巴（`div` 的 `#DE`、`cmpxchg16b` 的 16 字节原子、x87 的 80 位）在两边都还开着。

## 读上游上一轮（`a6967446..fbe249f6`）：44 条提交，CPU 侧 3 条

| 提交 | 内容 | 能不能吸收 |
|---|---|---|
| `bb1addf2` | `cpu: lower VEX.128/256 AVX horizontal and addsub FP ops in cpu_avx_horiz_fp_aot (#2629)`——`VHADDPS/PD`、`VHSUBPS/PD`、`VADDSUBPS/PD`，403 行升举器 + 1,229 行测试 | **部分吸收**。水平加/减本仓库已有（`VEC_HADD`，`vhsubps` 上一轮做过）；`VADDSUBPS/PD` 缺，按 SDM 补上（偶数通道**减**），片段差分判过一次 0/4 才对齐 |
| `780c0bc4` | `cpu: lower 32/64-bit BMI2 BZHI, MULX, PDEP, and PEXT in cpu_bmi2_bit_aot (#2621)`——688 行 + 1,898 行测试 | **部分吸收**。`bzhi`/`mulx` 早就有；`pdep`/`pext` 在这个映像的拒绝面里**一次都没出现**（不到 3 个函数），所以记在这里、不写代码 |
| `d371ce46` | `cpu: lower 16 VEX.256 AVX2 cross-lane extend and dword/qword unpack forms (#2614)`——`cpu_avx_int_aot.py` +136 行 | **已覆盖**。`VEC_EXTEND`（`pmovsx*`/`pmovzx*` 十二种）与 `VEC_UNPACK` 都在上一轮做过 |

其余 41 条是 shader / startup / d3d12 / CI / 测试，没有指令语义。

值得一提的两点对照：

- 上游把 `VHADD`/`VADDSUB` 这一类拆成一个 403 行的专用模块，本仓库仍是 `lift.py` 里的两张表
  （`VEC_HADD`、`VEC_ADDSUB`）加一个 `vec_line` 分支。**拆分的判据**还是那句：一张表能表达的
  就不该是一个模块；`vaddsubps` 与 `vaddps` 只差"偶数通道反向"，它属于表，不属于文件。
- 上游这一轮给 BMI2 配了 1,898 行测试，我们对应的是"一条指令一个函数"的片段差分（154 条）——
  同样的想法（**每条指令单独可证**），落点不同：他们比的是自己两套实现的输出，我们比的是硬件。

## 读上游上一轮（`e4e2b46d..a6967446`）：只有一条提交，且不在 CPU 侧

| 提交 | 内容 | 能不能吸收 |
|---|---|---|
| `a6967446` | `cpu: harden atomic writes, path validation, and subprocess error handling in exe_aot with CLI suite (#2605)`——`exe_aot.py` 改成 `os.replace` 原子落盘、路径校验、子进程错误传播，附带约 750 行的 CLI 测试套件 | **没有指令语义**。原子落盘这一条本仓库早就在做：差分前先写一份 `snippets.exe` 副本、从不改原件（`tools/diff.py` 的 `snippets()`），正是同一个原因——半写好的文件会让下一次运行拿到一个既不是原版也不是补丁版的映像 |

值得一提的对照：它那一轮把力气花在 **CLI 的健壮性**上，我们的对应物是 `diff.py` 的
`--keep` 与逐例故障恢复——**"裁判自己不能先坏"** 是两棵树共同的结论，只是落点不同。

上游按指令族拆成十几个 `cpu_*_aot.py`，本仓库只有一个 `lift.py`。本轮 `vec_line` 又长出
`VEC_ADDSUB`/`VEC_MUL`/`VEC_ABS`/`VEC_GATHER`/`VEC_MASKMOVE`/`VEC_VSHIFT`/`VEC_MINMAX` 七张表，
**拆分的判据已经压到桌面上**（见 README §7 末段）。
