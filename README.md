# Box Counter (Pallet Analyzer)

## Описание

"Box Counter" — единоразовое приложение для автоматического подсчёта коробок на паллетах по изображениям с двух сторон (left / right).

## Логика обработки изображений

1. **Данные**

   - В папке `images/` должны быть подпапки (pallet_1, pallet_2, ...), в каждой из которых:
     - `left.png`
     - `right.png`

2. **Модель YOLOv8**

   - Загружается `best.pt` для поиска:
     - `*_top` — верхушки коробок
     - `*_left` — лицевые стороны коробок
     - `pallet` — паллета

3. **Фильтрация**

   - Удаляются повторяющиеся объекты со значительным перекрытием масок (IoU > 0.8).

4. **Связывание в цепочки**

   - Ищем `*_top`, потом вниз по Y-оси добавляем `*_left` по условию попадания в область.

5. **Фильтр видимых коробок**

   - Сначала обрабатываем **left.png**, учитываются ближние коробки к левому нижнему углу.
   - Потом добавляем коробки с right.png, также ближние (по нижнему Y). Это позволяет избежать дублирования дальних коробок, которые могут быть не видны.
   - Работает сейчас криво, можно доработать в дальнейшем
---

## Отчёт

Результат выводится в `results/result.csv`

| directory | laptop | tablet | group_box | pallet |
|-----------|--------|--------|-----------|--------|
| pallet_1  | 3      | 0      | 1         | YES    |
| pallet_2  | 10     | 0      | 0         | YES    |
| pallet_3  | 0      | 0      | 0         | NO     |

---

## Запуск (Docker)

1. Сборка:

```bash
docker-compose build
```

2. Запуск:

```bash
docker-compose up
```

После этого в терминале выведется отчёт, и все результаты появятся в папке `results/`

---

## Глобальные переменные

- `image_root` — путь к папке с изображениями (`images/`)
- `output_root` — путь для сохранения масок (`results/images/`)
- `report_path` — путь к CSV отчёту (`results/result.csv`)
- `model` — загруженная модель YOLO с весами `best.pt`
- `class_names` — список классов из модели

---

## Инструкция по обучению

Для переобучения или добавления классов заказчиком:

### Требования:

- Python >= 3.9, я делал на 3.11
- torch, ultralytics
- видеокарта (опционально) с поддержкой CUDA 12.6 или другой версии (например в моем случае, RTX 4070 Super)

### Среда

- В **PyCharm** среда создаётся автоматически при запуске (venv).
- В **VS Code** необходимо вручную создать и активировать виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate  # для Linux/Mac
.venv\Scripts\activate     # для Windows
```

### Обучение на GPU

```bash
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Убедитесь, что используется PyTorch с поддержкой CUDA 12.6 или нужной вам версии.

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=your_data.yaml epochs=50 device=0
```

### Обучение на CPU

```bash
pip install ultralytics
```

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=your_data.yaml epochs=50
```

Файл `your_data.yaml` должен содержать:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 3  # number of classes
names: ["laptop_top", "tablet_top", "group_box_top", ...]
```

---

Описание датасета

Датасет используется для задачи сегментации в формате YOLOv8. Структура:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```


Файл data.yaml:
```
train: dataset/images/train  # Путь к тренировочным данным
val: dataset/images/val      # Путь к валидационным данным
test: dataset/images/test    # Путь к тестовым данным
nc: 7                        # Количество классов
names: ["laptop_top", "tablet_top", "group_box_top", "laptop_left", "tablet_left", "group_box_left", "pallet"]
```

Уже готовый датасет можно найти в директории dataset, либо же перейдя по [ссылке](https://app.roboflow.com/dataset-o4eny/boxes-bxdaw/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true). В данном случае было около 70 изначальных фотографий, аугментировано до 140 (этого достаточно, но при необходимости можно увеличить).

Если ссылка на датасет недоступна, то можно мне написать в телеге (@Karkush), перешлю еще раз. 


---

## Рекомендации по улучшению распознавания

Было бы полезно модернизировать упаковки для классов `laptop` и `tablet`:

- Использовать разные цвета краёв коробок
- Применить уникальные узоры для разных классов


Это особенно важно для дальних коробок, где слабый контраст затрудняет распознавание.

Также советую доделать, либо переделать последний пункт из логики работы. Я делал всё в 1 день, ибо на носу пасха, а у вас, думаю, времени будет больше.

---

## Результат

После выполнения:

- Сохранённые маски объектов в `results/images/`
- CSV-отчёт `results/result.csv`
- Вывод в терминал


---

