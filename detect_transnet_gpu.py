import os
import sys
from datetime import datetime, timedelta
import numpy as np
import ffmpeg
import torch
from scipy.signal import find_peaks
from tqdm import tqdm

# Добавляем путь к PyTorch-версии TransNetV2
sys.path.append('TransNetV2/inference-pytorch')
from transnetv2_pytorch import TransNetV2

# --- 1. КОНФИГУРАЦИЯ ---
VIDEO_PATH = os.path.join('videos_processed', 'video_compilation_021.mp4')
CHAPTERS_PATH = os.path.join('videos_chapters', 'video_compilation_021.txt')
PYTORCH_WEIGHTS_PATH = os.path.join('TransNetV2', 'inference-pytorch', 'transnetv2-pytorch-weights.pth')
# Папка для сохранения результатов
OUTPUT_DIR = 'transnet_results'
TOLERANCE_SECONDS = 1.0
BATCH_SIZE = 1000

# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def parse_chapters_file(file_path: str) -> list[float]:
    timestamps_sec = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 2: continue
                time_str = parts[1]
                main_time, ms_part = time_str.rsplit(':', 1)
                dt_obj = datetime.strptime(main_time, '%H:%M:%S')
                total_seconds = (dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second + int(ms_part) / 100.0)
                timestamps_sec.append(total_seconds)
    except FileNotFoundError:
        print(f"Ошибка: Файл с главами не найден: {file_path}")
        return []
    return [ts for ts in timestamps_sec if ts > 0]

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    peaks, _ = find_peaks(predictions, height=threshold, distance=1)
    if len(peaks) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    scene_boundaries = np.concatenate(([0], peaks, [len(predictions) - 1])).astype(np.int32)
    unique_boundaries = sorted(list(set(scene_boundaries)))
    
    scenes = np.array([
        [unique_boundaries[i], unique_boundaries[i+1] -1]
        for i in range(len(unique_boundaries) - 1)
    ], dtype=np.int32)
    
    return scenes

# ✅ НОВАЯ ФУНКЦИЯ для форматирования времени
def format_seconds(seconds: float) -> str:
    """Конвертирует секунды в строку формата HH:MM:SS:ms."""
    td = timedelta(seconds=seconds)
    minutes, sec = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = int(td.microseconds / 10000) # 2 знака для ms
    return f"{hours:02d}:{minutes:02d}:{sec:02d}:{milliseconds:02d}"

# --- 3. ОСНОВНОЙ СКРИПТ ---
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Ошибка: Видеофайл не найден: {VIDEO_PATH}")
        return
    if not os.path.exists(PYTORCH_WEIGHTS_PATH):
        print(f"Ошибка: Файл с весами PyTorch не найден: {PYTORCH_WEIGHTS_PATH}")
        return

    # ✅ ИЗМЕНЕНИЕ: Отображение используемых файлов
    print(f"📖 Используемый файл видео:    {VIDEO_PATH}")
    print(f"📖 Используемый файл глав: {CHAPTERS_PATH}")

    print("\n🔍 Получение информации о видео...")
    try:
        probe = ffmpeg.probe(VIDEO_PATH)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        framerate_str = video_info['r_frame_rate'].split('/')
        framerate = float(framerate_str[0]) / float(framerate_str[1])
        num_frames = int(video_info['nb_frames'])
        print(f"Видео: {video_info['width']}x{video_info['height']}, {num_frames} кадров, {framerate:.2f} fps.")
    except Exception as e:
        print(f"Не удалось прочитать информацию о видео: {e}")
        return

    print("\n🚀 Инициализация PyTorch-модели TransNetV2...")
    model = TransNetV2()
    model.load_state_dict(torch.load(PYTORCH_WEIGHTS_PATH))
    model.eval().cuda()

    print("\n🚀 Начало декодирования видео с GPU-ускорением и предсказания...")
    
    TARGET_WIDTH, TARGET_HEIGHT = 48, 27
    args = (
        ffmpeg
        .input(VIDEO_PATH, hwaccel='cuda', vsync=0)
        .filter('scale', width=TARGET_WIDTH, height=TARGET_HEIGHT)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args('-hide_banner', '-loglevel', 'error')
    )
    process = ffmpeg.run_async(args, pipe_stdout=True)

    all_predictions = []
    
    with tqdm(total=num_frames, unit="frames", desc="Processing video") as pbar:
        with torch.no_grad():
            while True:
                in_bytes = process.stdout.read(TARGET_WIDTH * TARGET_HEIGHT * 3 * BATCH_SIZE)
                if not in_bytes: break
                
                in_frames = np.frombuffer(in_bytes, np.uint8).reshape([-1, TARGET_HEIGHT, TARGET_WIDTH, 3]).copy()
                in_frames_tensor = torch.from_numpy(in_frames).cuda().to(torch.uint8)
                single_frame_pred, _ = model(in_frames_tensor.unsqueeze(0))
                
                predictions = torch.sigmoid(single_frame_pred).cpu().numpy()
                all_predictions.append(predictions)
                pbar.update(len(in_frames))

    process.wait()
    print("\n✅ Декодирование и предсказание завершены.")
    
    final_predictions = np.concatenate(all_predictions, axis=1).flatten()
    
    scenes = predictions_to_scenes(final_predictions)
    detected_frames = [scene[0] for scene in scenes if scene[0] > 0]
    
    print(f"✅ Найдено {len(detected_frames)} переходов.")

    # ✅ ИЗМЕНЕНИЕ: Сохранение результатов в файл
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, frame_num in enumerate(detected_frames, 1):
            timestamp_sec = frame_num / framerate
            time_str = format_seconds(timestamp_sec)
            line = f"{i:07d} {time_str} {frame_num}\n"
            f.write(line)
    
    print(f"💾 Результаты сохранены в файл: {output_path}")

    print("\n📊 Начало оценки результатов...")
    
    ground_truth_times = parse_chapters_file(CHAPTERS_PATH)
    detected_times = [frame / framerate for frame in detected_frames]

    if not ground_truth_times or not detected_times:
        print("Нет данных для разметки или детекции. Оценка невозможна.")
        return

    print(f"Загружено {len(ground_truth_times)} истинных переходов.")
    print(f"Обнаружено {len(detected_times)} потенциальных переходов.")

    true_positives, false_negatives = 0, 0
    temp_detected_times = list(detected_times)
    for gt_time in ground_truth_times:
        found_match, best_match, min_diff = False, -1, float('inf')
        for det_time in temp_detected_times:
            diff = abs(gt_time - det_time)
            if diff <= TOLERANCE_SECONDS and diff < min_diff:
                min_diff, best_match, found_match = diff, det_time, True
        if found_match:
            true_positives += 1
            temp_detected_times.remove(best_match)
        else: false_negatives += 1
            
    false_positives = len(detected_times) - true_positives
    recall = true_positives / len(ground_truth_times) if ground_truth_times else 0.0
    precision = true_positives / len(detected_times) if detected_times else 0.0

    print("\n--- [ОТЧЕТ TransNetV2 c PyTorch] ---")
    print(f"Найдено верных переходов (True Positives): {true_positives}")
    print(f"Пропущено истинных переходов (False Negatives): {false_negatives}")
    print(f"Лишние (ложные) переходы (False Positives): {false_positives}")
    print("-------------------------")
    print(f"🎯 Полнота (Recall):    {recall:.2%}")
    print(f"📈 Точность (Precision): {precision:.2%}")
    print("-------------------------")

if __name__ == '__main__':
    main()