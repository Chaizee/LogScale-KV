# Multi-Scale Streaming KV-Cache (MS-KV) — Исследовательский прототип

Прототип для проверки и улучшения идеи из черновика *Streaming MERA-KV*.
Написан на **чистом PyTorch** без CUDA-only расширений (`flash-attn`, `bitsandbytes`, CUDA Triton), поэтому запускается на **AMD GPU** (ROCm на Linux/Windows, DirectML на Windows) и CPU.

---

## Что улучшено по сравнению с исходным черновиком

1. **Убрана ложная «квантовая» обёртка (честная математика):**
   Вместо псевдо-MERA поверх конкатенаций реализован **потоковый многомасштабный диадный кэш (Exponential-Histogram Tree)** с настраиваемыми операторами слияния.
2. **Устранён «обрыв разрешения» на степенях двойки (`k`-слотовый избыточный буфер):**
   В оригинальном Carry-Bit (`k = 1`) при $N_{\text{hist}} = 2^m$ все младшие уровни обнуляются и недавние токены схлопываются с токеном №1. Замена на `k >= 2` слотов на уровень (сливаются 2 старейших блока при $n > k$) гарантирует, что на каждом активном уровне всегда остаётся $k-1 \dots k$ блоков.
   Кроме того, реализовано **векторизованное $O(\log N)$ обновление** (`m = (n - k + 1) // 2`), бит-в-бит эквивалентное потоковому 1-by-1 обновлению и пригодное для параллельного prefill и обучения.
3. **Исправлен барицентрический RoPE (Аналитическое затухание высоких частот):**
   Для блока из $L$ токенов точное среднее повёрнутых ключей равно $\gamma_j(L) \cdot R_{\Theta, \bar{m}} k_0$, где $\gamma_j^{\text{dir}}(L) = \frac{\sin(L\theta_j / 2)}{L\sin(\theta_j / 2)}$, а гладкая гауссова аппроксимация $\gamma_j^{\text{gauss}}(L) = \exp\!\left(-\frac{(L^2 - 1)\theta_j^2}{24}\right)$ совпадает с ним до $O(\theta^2)$ и подавляет фазовый шум высокочастотных компонент RoPE.
4. **Исправлен взрыв норм $2^{l/2}$ и добавлен обучаемый `GatedCayleyMerger`:**
   Вместо ненормированной изометрии $W^T W = I$ (растящей коррелированный сигнал в $2^{l/2}$ раз) используется выпуклый input-dependent gate $g \in (0, 1)^D$ + ортогональное преобразование Кэли $Q_d \in O(D)$ на разностной моде $(x_0 - x_1)$. Свойства:
   - строго сохраняет постоянный сигнал ($f(u, u) = u$ при любых весах — нет взрыва норм);
   - инициализируется в точности как `0.5 * (x0 + x1)` (стабильный старт обучения);
   - поддерживает RG Fixed-Point weight sharing для $l \ge l_{\text{crit}}$.
5. **Гибридная 4-зонная топология для спасения NIAH / RULER:**
   `Attention Sinks + Salient Pool (тяжёлые/аномальные токены) + k-slot Multi-Scale Tree + Local Window`. Редкие «иголки» защищаются от размывания, а остальной контекст не удаляется, а сжимается в логарифмическое дерево.

---

## Установка на ПК с AMD видеокартой

### Вариант A: Linux + AMD ROCm (рекомендуется, максимальная скорость)
```bash
python -m venv .venv
source .venv/bin/activate
# Выберите актуальную версию ROCm под вашу карту (пример для ROCm 6.2):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```
*(PyTorch под ROCm использует устройство `cuda` — скрипт распознает его автоматически через `--device auto`.)*

### Вариант B: Windows + AMD GPU (DirectML или ROCm on Windows)
Если на Windows стоит ROCm PyTorch — используйте `pip install` под ROCm. Если обычный Windows без ROCm:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torch-directml
pip install -r requirements.txt
```
*(При DirectML скрипт сам выберет `directml` и безопасный `fp32`; при желании можно попробовать `--dtype fp16`.)*

---

## Порядок запуска экспериментов

### Шаг 1. Быстрая математическая проверка (CPU, 2 секунды, без скачивания моделей)
Проверяет эквивалентность потокового и чанкового алгоритмов, воспроизводит обрыв Carry-Bit и взрыв $2^{l/2}$, проверяет точность формулы Dirichlet-RoPE и ортогональность `GatedCayleyMerger`:
```bash
python tests/sanity_check.py
```

### Шаг 2. Полная матрица абляций (Training-Free, ~3–10 минут)
Запустит 10 конфигураций на `Qwen/Qwen2.5-0.5B-Instruct` (открытая модель с GQA + RoPE, не требует HF-токена) и выведет сравнительную таблицу (PPL, NIAH, Multi-Key, сжатие, скорость, VRAM):
```bash
python benchmarks/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --task all --seq-len 2048 --local-size 128 --salient-size 64 --slots-per-level 4
```
Для проверки на более серьёзной длине и модели (если у AMD-карты 12–24 ГБ VRAM):
```bash
python benchmarks/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --task all --seq-len 4096 --local-size 256 --salient-size 64 --slots-per-level 4
```
Если на ПК нет доступа к HuggingFace Datasets (или хотите прогнать на своей книге/тексте), передайте `--text-file path/to/book.txt`.

### Шаг 3. Диагностика «обрыва разрешения» на степенях двойки
Сравнивает пер-токенный NLL до и после $N_{\text{hist}} = 512$ для 1-слотового Carry-Bit против 4-слотового дерева:
```bash
python benchmarks/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --task cliff --seq-len 1024 --local-size 128 --cliff-pow2 512
```

### Шаг 4. Дистилляция обучаемого `GatedCayleyMerger` (опционально)
Быстрая послойная дистилляция на замороженном бэкбоне (помещается в 8 ГБ VRAM):
```bash
python train_distill.py --model Qwen/Qwen2.5-0.5B-Instruct --train-mode layerwise --prefix-len 1536 --suffix-len 256 --steps 60
```
Затем прогон обученного мерджера в общей таблице абляций:
```bash
python benchmarks/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --task ablation --seq-len 2048 --merger-ckpt results/gated_cayley_merger.pt
```

---

## На что смотреть в результатах (критерии жизнеспособности идеи)

1. **Строка 4 (`StiefelUnscaled`) vs Строка 5 (`Mean`):** подтвердит, действительно ли ненормированная изометрия $W^T W = I$ ломает PPL из-за роста норм $2^{l/2}$.
2. **Строка 5 (`Carry-Bit k=1 + BaryRoPE`) vs Строка 6 (`MS-Tree k=4 + GaussDampedRoPE`):** покажет чистый вклад наших двух математических исправлений (отсутствие клиффа + затухание высоких частот RoPE).
3. **Строка 6 (`GaussDamped`) vs Строка 7 (`PostRoPE`) vs Строка 8 (`No LogBias`):** даст честный ответ, работает ли `pre_rope_damped_gauss` лучше простейшего `post_rope` и даёт ли $+l\ln 2$ реальный выигрыш на предобученной LLM.
4. **Строка 6 (`MS-Tree`) vs Строка 2 (`StreamingLLM`) vs Строка 3 (`H2O-Lite`) vs Строка 10 (`Hybrid MS-KV`):** ключевой тест для статьи. Если **Hybrid MS-KV** на том же бюджете слотов одновременно держит **PPL лучше StreamingLLM/H2O-Lite** и **NIAH/MultiKey на уровне H2O-Lite/Full**, у вас на руках сильный, доказуемый научный результат.
