# 静态重编译研究：Xbox Series X|S → Windows x64

> 仓库代号 `ZJOU-JWXT-PC`。

把一个 x86-64 主机镜像里的**机器码离线翻译成可编译的 C**，再由宿主编译器生成原生可执行文件。不解释、不在运行时翻译二进制；原文的语义由**硬件本身**做裁判。

本仓库只有工具与文档，**不含任何游戏数据、密钥或反编译产物**。使用者需自备依法获得的副本。

<p align="center">
  <img alt="coverage" src="https://img.shields.io/badge/%E5%8D%87%E4%B8%BE%E8%A6%86%E7%9B%96-35.8%25%20(%E5%89%8D4%E4%B8%87%E5%87%BD%E6%95%B0)-blue">
  <img alt="oracle" src="https://img.shields.io/badge/%E7%A1%AC%E4%BB%B6%E5%B7%AE%E5%88%86-244%20%E4%BE%8B%20%2F%2093.4%25-brightgreen">
  <img alt="functions" src="https://img.shields.io/badge/%E5%87%BD%E6%95%B0%E8%A1%A8-315%2C990-informational">
</p>

---

## 1. 结论先行

| 问题 | 答案 |
|---|---|
| 这条路能省掉什么？ | **省不掉"翻译"**——guest 与 host 同为 x86-64，没有翻译红利。买到的是：可编辑的代码、整体替换图形边界、消掉三份内存 |
| 现在做到哪一步？ | 前 40,001 个函数里 **14,330（35.8%）** 能翻成可编译的 C；函数表已长到 **315,990** 个起点 |
| 翻译对了没有？ | 用**原始机器码直接 `call`** 做裁判：400 个函数里 276 个进得了差分，其中 121 个跑出 244 例，**228 例（93.4%）逐寄存器逐字节一致** |
| 最缺什么？ | **环境**，不是翻译。149 个函数在两侧同样崩溃——它们读的全局在文件里是 0，只有游戏自己的初始化代码才会写 |

---

## 2. 方法：四步

```
PE 镜像 ──① 取边界 ──► 函数表 ──② 升举 ──► C 源码 ──③ 编译 ──► .exe
                    (.pdata)         (capstone)      (cl.exe)
                                          │
                                          └──④ 裁判：原始字节 call 过去，读回来比
```

1. **取边界**：`.pdata` 的 `RUNTIME_FUNCTION` 数组（12 字节：起/止/展开信息）免符号表给出 299,257 个函数，覆盖 `.text` 的 92%。剩下 8% 是**没有展开信息的叶子函数**，用 `call` 目标反推补全——目标就是函数的第一条指令，迭代到收敛，长到 **315,990**。
2. **升举**：每个函数 `void f(uint64_t *r)`，`r[0..15]` 是 rax..r15。操作数按自身宽度读写（写 `esi` 清零高半、写 `al` 保位、`ah` 是偏移 8）；`[rip+d]` 在升举期就折成 `img + rva`；调用另一个升举函数**不需要 marshalling**，因为两边共用同一个寄存器文件。`call` 会先把返回地址写进 `rsp-8` 并把 `rsp` 降 8——和硬件逐字节一致，否则被调函数建的整个帧都偏一格。
3. **编译**：MSVC。生成的是普通 C，不是 IR——能不能过编译器是最低门槛。
4. **裁判**：同 ISA 意味着原实现不需要模拟器。**guest 指针就是 host 指针**：映像映射在它的首选基址、arena 映射在固定的 `0x20_0000_0000`，于是 `mov [rcx], rax` 两侧都是"存到 rcx 指向的地方"，比较的是 16 KiB arena 而不是一个折叠地址的临时缓冲区。95.5% 的升举函数含 `call`，所以未升举的目标走 [`guest_bridge`](tools/oracle/src/lib.rs)——和裁判同一个调用桩——把**被测函数**换成 C，它下面调用的仍是游戏自己的代码。

---

## 3. 已实现

| 工具 | 作用 | 实测 |
|---|---|---|
| `tools/probe.py` | PE 只读情报：函数数、代码量、导入面 | 299,257 函数 / 95.25 MB `.text` / 中位 117 B |
| `tools/lift.py` | 直线段升举 + 自检解释器 | 子寄存器、`movabs`、`[rip+d]`、映像加载器 |
| `tools/branch.py` | **按边遍历**的 CFG + 分支 + 跳转表 | 跳转表 → `switch` |
| `tools/diff.py` | 与硬件差分（真实地址空间 + 桥） | 244 例中 228 一致 |
| `tools/oracle/` | 硬件裁判（Rust，内联汇编）+ `guest_bridge` cdylib | 自检 `rax=42`；映像与 arena 两侧同址 |
| `tools/callgraph.py` | 调用图统计 | 76.8% 会调用，18% 间接调用 |
| `tools/functable.py` | 用 `call` 目标生长函数表 | +16,733 个起点 |
| `tools/closure.py` | 闭包升举成一份 C | 未覆盖目标落 `miss()` |
| `tools/cover.py` | 接受量 + 拒绝面直方图 | 本页所有覆盖数字的来源 |

### 覆盖演进

| 阶段 | 接受量（前 40,001 函数） | 关键改动 |
|---|---|---|
| 窥孔 `cmp/test + jcc` | 9,331（23.3%） | 起点 |
| 子寄存器 + `movabs` + `[rip+d]` | 12,723（31.8%） | 操作数宽度、映像映射 |
| ALU 结果供 ZF/SF、`BT` 供 CF | 13,882（34.7%） | 生产者不止 `cmp` |
| **CFG 遍历 + 跳转表 → `switch`** | **14,330（35.8%）** | 不再线性扫描 |

### 拒绝面（同一批函数）

| 原因 | 数量 | 性质 |
|---|---|---|
| 间接调用 | 7,861 | 虚表/导入；静态可解性有限，需要运行时派发表 |
| `vmovaps` / `vmovss` / `vxorps` | 9,947 | SIMD：机械工作，但要连 `xmm` 一起进差分 |
| `segment` 前缀 | 1,029 | 多为 `gs:` 的 TEB/TLS 访问 |
| `div` / `lock cmpxchg` | 1,324 | 需要真正的除法与原子读改写的标志语义 |
| 间接跳转 | 584 | 虚表尾调用、导入 thunk（不是跳转表，正确拒绝） |

---

## 4. 三条被数据否定的路线

写在这里，是因为它们看起来都更省事。

**① 单函数独立成立。** 分支升举器只接受 121 / 207,811——墙不是分支，是调用。一个函数调另一个函数，单函数独立这条路本身就走不通；把两边一起翻，`call` 就变成 C 里的直接调用。

**② 把上游 82 个提交 cherry-pick 进来。** 结果是 **82/82 全冲突**，而且不是"冲突多"：本树把 `bindless.rs` 拆成 `descriptor/*`，上游对它的改动是 delete/modify（0 个冲突块）；其余改动各自挂在它自己的前置设施上（每堆写标记、内存记账）。**结构上不可移植。**

**③ 运行时翻译（HLE）路线的账本。** 世界场景稳态 5.0 fps = 200.7 ms/帧，其中命令录制 149.1 ms、主机内存↔宿主副本 86.8 ms。这些在主机上是 0——统一内存 + 硬件直接消费绘制包。这就是为什么要走到编译期。

---

## 5. 验证：从 2.5% 到 69%，剩下的墙是环境

`14,330` 是**翻译量**。能被证伪的才是**验证量**，两者之间隔着"这个函数跑不跑得起来"。

| 指标 | 上一版 | 现在（`--limit 400 --skip 6000 --branches --rounds 2`） |
|---|---|---|
| 升举成功 | 148 | 400 |
| 进得了差分（能在 arena 里跑完） | **10（2.5%）** | **276（69%）** |
| 真跑出了可比结果 | 10 | **121 个函数 / 244 例** |
| 逐寄存器 + 逐字节一致 | 30/30 | **228/244（93.4%）** |

原因一句话：内存模型原来把每个访问都当成那 256 字节栈，`mov [rcx], rax` 就是越界，于是构造函数、
setter 这类"写调用方给的缓冲区"的函数全被滤掉。现在 guest 地址就是 host 地址，这个过滤条件消失了。

**剩下的 149 个函数在两侧同样崩溃**，而这才是一道真正的墙：

```
mov rax, qword ptr [rip + 0xa9b2bf7]   ; 这个全局在文件里是 0
mov ecx, dword ptr [rax + 0xbf4]       ; rax = 0，直接 AV
```

它们读的是游戏初始化代码才会写的全局。裁判把映像按文件布局映射、没有跑过 `main`，所以这些槽位全是 0。
不是翻译错了，是**没有环境**——单函数独立成立这条路到此为止，再往上只能"整份链接后跑入口"（见[路线图](docs/roadmap.md) M1.5）。

另外 6 个只在移植侧崩溃、16 例结果不一致，已定位的方向有三类：IAT 里解析出的函数地址**跨进程不同**
（被存进 arena 就成为差异）、未初始化全局导致走不同分支、以及若干 32 位写零扩展的边界。

### 桥：唯一的"作弊点"，也是唯一的可行点

95.5% 的升举函数含 `call`。如果裁判只能进入叶子函数，可测比例的上限就是 0.5%——实测 20,000 个函数里
只有 30 个不含 `call`。所以未升举的目标通过 `guest_bridge` 调回原始字节：

```c
void f_0x4f862d0(uint64_t *r) {          /* 未升举：走桥 */
    r[4] += 8;                            /* 升举侧已经压过返回地址，先还原 */
    note(r[4] - 8);                       /* 桥会在这里压一个宿主地址，记下来 */
    guest_bridge(r, img + 0x4f862d0);
    r[4] -= 8;
}
```

桥和裁判是**同一份内联汇编**（`tools/oracle/src/lib.rs` 的 `call_guest`），所以被测函数之上是 C、
之下是原码。唯一对不上的只有那个 8 字节槽：硬件压的是映像地址，桥压的是宿主地址。移植侧把槽位报给
裁判、裁判跳过它——不是"差不多就行"，是**精确地少比一个已知会不同的 8 字节**。

## 6. 复现

```powershell
python tools/probe.py  <游戏目录>\game_scarlett_master.exe     # 只读情报
python tools/cover.py  --sample 40000                          # 覆盖与拒绝面
python tools/diff.py   --limit 400 --skip 6000 --branches --rounds 2   # 与硬件差分（含桥）
python tools/closure.py --root 0x204cb0 --out out/closure.c --compile
```

依赖：Python 3.11+（`capstone`、`pefile`）、MSVC（`cl.exe`）、Rust（`tools/oracle`）。
`diff.py` 用 `--oracle` / `--bridge` 指向裁判与桥，默认在 `D:\zjou-build\release\`。
差分前先构建裁判：

```powershell
cd tools/oracle; cargo build --release    # oracle.exe + oracle_guest.dll（桥）
```

---

## 7. 参考与引用

两条关键规则不是我们发明的，来自目前唯一公开自述"x64 机器码 → C++ 静态重编译"并持续开发的项目：

1. **Xbox2Windows**（R0Y00，MIT）— <https://github.com/R0Y00/Xbox2Windows>
   - 标志位只在"生产者 → 消费者、且该直线段无其他入边"的范围内建模；`Jcc`/`SETcc`/`CMOVcc` 是消费者，`mov`/`lea`/`not`/`bswap`/`push`/`pop`/`cbw`/`cwde`/`cdqe`/`cqo` 不破坏标志。见其 `tools/cpu_switch_aot.py` 的模块注释。
   - 跳转表只接受两种形态，且基址必须在每条路径上是**同一个 RIP 相对地址**、索引必须被证明**等于参与比较的值**；`BT/BTS/BTR/BTC` 的位偏移对寄存器基址取模到操作数宽度。
   - **只顺着边遍历，不线性扫描**——MSVC 把跳转表放在 `jmp` 之后，线性扫描会把表本身当成代码。本仓库 `tools/branch.py` 的 `walk()` 即此规则。
2. **XenonRecomp**（hedge-dev，MIT）与 **N64Recomp** — 闭包式重编译的先行者：函数边界从调用边反推、跳转表转 `switch`、"翻译整份代码再链接"而不是逐函数独立成立。
3. **VIctory**（Marvisak，GPLv3）、**XCL**（ivall）、**Palmetto**（luizwidmer）— 运行时翻译（HLE）路线的对照实现。本仓库**未使用其代码**；§4③ 的性能账本来自对它们的实测（另：同一台 RTX 4080 Laptop 上，VIctory 于第 194 帧设备移除，四个 fork 存活——说明那是该实现的问题，不是平台限制）。
4. **Intel® 64 and IA-32 Architectures Software Developer's Manual, Vol. 2** — 指令语义与标志位定义的最终依据（尤其 `BT`、移位计数的掩码、子寄存器的清零规则）。
5. **Microsoft PE/COFF Specification** — `.pdata` / `RUNTIME_FUNCTION` 布局。解析工具：`capstone`、`pefile`。

各参考仓库的阅读笔记见 [`docs/reference-xbox2windows.md`](docs/reference-xbox2windows.md) 与 [`docs/reference-clones.md`](docs/reference-clones.md)。

---

## 8. 文档

| 文件 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 分层与模块边界 |
| [`docs/roadmap.md`](docs/roadmap.md) | 里程碑与验收标准（含当前阻塞项） |
| [`docs/reference-xbox2windows.md`](docs/reference-xbox2windows.md) | 对上游做法的采纳/差异清单 |
| [`docs/reference-clones.md`](docs/reference-clones.md) | 两个参考仓库的结构与可读部分 |

## 许可

MIT。参考项目各自许可见 §7；本项目不包含、不分发任何游戏数据。
