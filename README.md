# Детектор смены сцен на базе TransNetV2 (GPU) 🚀

Этот проект использует PyTorch-реализацию модели **TransNetV2** для высокоскоростного обнаружения смены сцен в видеофайлах с использованием NVIDIA GPU (CUDA). Скрипт обрабатывает видео, находит кадры, где происходит смена сцены, и сохраняет их временные метки.

Проект основан на репозитории: [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2).

## 🎬 Ключевые возможности

* **GPU-ускорение**: Использует CUDA для декодирования видео и инференса модели, что значительно ускоряет обработку.
* **Высокая точность**: Применяет предобученную модель TransNetV2, известную своей эффективностью в задачах детекции границ сцен.
* **Оценка качества**: Автоматически рассчитывает метрики **Recall** (полнота) и **Precision** (точность), если предоставлен файл с истинными временными метками (главами).
* **Простота использования**: Основная логика заключена в одном Python-скрипте.

## 📁 Структура проекта

Для корректной работы скрипта ваш проект должен иметь следующую структуру:

```
/Detect-TransNetV2-GPU/
│
├── TransNetV2/                  # <-- Клон репозитория soCzech/TransNetV2
│   └── inference-pytorch/
│       └── transnetv2-pytorch-weights.pth  # <-- Веса модели
│
├── videos_processed/            # <-- Папка для ваших видео
│   └── video_compilation_021.mp4
│
├── videos_chapters/             # <-- Папка для файлов с главами (для оценки)
│   └── video_compilation_021.txt
│
├── transnet_results/            # <-- Папка для сохранения результатов (создается автоматически)
│
├── detect_transnet_gpu.py       # <-- Основной скрипт
└── requirements.txt             # <-- Файл с зависимостями
```

## 🛠️ Установка и настройка

### Шаг 1: Системные требования

1.  **Оборудование**: **NVIDIA GPU** с поддержкой CUDA.
2.  **Программное обеспечение**:
    * **Python 3.8+**
    * **FFmpeg**: Установленный в вашей системе и доступный в PATH. Инструкции по установке для [Windows](https://phoenixnap.com/kb/ffmpeg-windows), [macOS](https://trac.ffmpeg.org/wiki/CompilationGuide/macOS), [Linux](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu).
    * **CUDA Toolkit**: Установленный и совместимый с вашей версией PyTorch.

### Шаг 2: Клонирование репозиториев

Сначала клонируйте этот репозиторий, а затем — репозиторий TransNetV2 **внутрь него**.

```bash
# Клонируем основной репозиторий
git clone [https://github.com/wolfam0108/Detect-TransNetV2-GPU.git](https://github.com/wolfam0108/Detect-TransNetV2-GPU.git)
cd Detect-TransNetV2-GPU

# Клонируем TransNetV2 ВНУТРЬ него
git clone [https://github.com/soCzech/TransNetV2.git](https://github.com/soCzech/TransNetV2.git)
```

### Шаг 3: Загрузка весов модели

Скачайте файл весов модели `transnetv2-pytorch-weights.pth` со страницы [релизов TransNetV2](https://github.com/soCzech/TransNetV2/releases) и поместите его по следующему пути:

`TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth`

### Шаг 4: Установка Python-зависимостей

Рекомендуется использовать виртуальное окружение.

```bash
# Создание и активация виртуального окружения (опционально, но рекомендуется)
python -m venv venv
source venv/bin/activate  # Для Linux/macOS
# venv\Scripts\activate   # Для Windows

# Установка зависимостей из файла requirements.txt
pip install -r requirements.txt
```

> **Примечание**: Убедитесь, что устанавливаемая версия `torch` совместима с вашей версией CUDA. Если у вас возникли проблемы, установите PyTorch вручную с [официального сайта](https://pytorch.org/get-started/locally/).

### Шаг 5: Подготовка данных

1.  Создайте папки `videos_processed` и `videos_chapters`.
2.  Поместите ваш видеофайл (например, `video_compilation_021.mp4`) в папку `videos_processed`.
3.  (Опционально) Если у вас есть файл с реальными временными метками для оценки точности, создайте его (например, `video_compilation_021.txt`) и поместите в `videos_chapters`.

## 📝 Пример файла с главами

Файл `video_compilation_021.txt` должен содержать временные метки смены сцен в формате `[Название] ЧЧ:ММ:СС:мс`. Скрипт считывает только временную метку.

**Пример содержимого `video_compilation_021.txt`:**

```
CHAPTER01 00:00:15:43
CHAPTER02 00:00:48:12
CHAPTER03 00:01:02:98
CHAPTER04 00:01:25:00
```

## ▶️ Запуск

1.  Убедитесь, что пути к видео и файлу глав в скрипте `detect_transnet_gpu.py` указаны верно:

```python
VIDEO_PATH = os.path.join('videos_processed', 'video_compilation_021.mp4')
CHAPTERS_PATH = os.path.join('videos_chapters', 'video_compilation_021.txt')
```

2.  Запустите скрипт:

```bash
python detect_transnet_gpu.py
```

## 📈 Результаты

* **В консоли** вы увидите прогресс обработки и итоговый отчет с метриками (True Positives, False Negatives, Recall, Precision).
* **В файле** `transnet_results/video_compilation_021.txt` будут сохранены все найденные временные метки смен сцен.