# Continuous OCR Learning

Сервис непрерывного обучения двух OCR-моделей по таблице в PostgreSQL:

* **words** — картинки с русскими словами;
* **digits** — картинки с числами.

Группа строки определяется по содержимому `answer`: если в нём только цифры — это
`digits`, иначе `words` (`config.classify_answer`).

## Как это работает

```
PostgreSQL (id uuid, image base64, answer)
        │  поиск новых строк каждые POLL_INTERVAL секунд
        ▼
  ingest: base64 -> PNG в локальный кэш var/data/<group>/
        │  счётчик новых строк по группам
        ▼
  >= 100 новых в группе -> дообучение этой модели (replay + откат при регрессии)
        ▼
  чекпоинт var/checkpoints/<group>.pt -> модель в памяти обновляется
        ▲
  HTTP /predict использует актуальные веса
```

* **Модель** — CRNN (CNN-энкодер + 2-слойный BiLSTM + CTC), своя на каждую группу,
  со своим алфавитом. Длина текста произвольная, сегментация символов не нужна.
* **Поиск новых строк.** `id` — uuid, поэтому «всё, что больше последнего id»
  не работает. Поддерживаются два режима:
  * `CL_CURSOR_COLUMN=created_at` (рекомендуется) — keyset-обход по паре
    `(created_at, id)`: курсор запоминается в `state.json`, запрос опирается на
    индекс `(created_at, id)`, строки с одинаковым временем не теряются;
  * колонка не задана — сервис читает список `id` из таблицы и вычитает из него
    то, что уже лежит в кэше, дочитывая только недостающие строки. Работает
    всегда, но каждый опрос читает все id таблицы.
* **Кэш датасета.** Картинка из базы декодируется один раз и кладётся на диск как
  PNG (`var/data/<group>/images/<id>.png`, метки — в `labels.jsonl`). Дообучение
  не ходит в базу за старыми строками.
* **Фильтр на приёме.** Ответ из одних цифр короче 5 символов в обучение не
  идёт — ни в `digits`, ни в `words`; порог задаёт `MIN_DIGITS_ANSWER_LENGTH`
  в `constants/training.py`. Id таких строк запоминаются рядом с битыми
  картинками, поэтому из базы они больше не вычитываются. Коротких слов правило
  не касается: «кот» обучению не мешает.
* **Разбиение train/val** детерминированное, по `crc32(id) % 1000` — один и тот же
  пример всегда в той же части, поэтому метрики сравнимы между версиями модели.
* **Дообучение (replay).** На каждый новый пример подмешивается `REPLAY_RATIO`
  старых — это держит катастрофическое забывание под контролем.
* **Откат.** После дообучения считается `val_cer`. Если он ухудшился больше, чем
  на `CER_REGRESSION_TOLERANCE`, веса откатываются к предыдущим, а чекпоинт не
  перезаписывается. Новые данные остаются в кэше и попадут в следующую порцию.
* **Роутер групп.** Маленький CNN-классификатор «слово или число» учится на тех
  же кэшированных данных и переобучается после каждого успешного обучения любой
  из OCR-моделей. Он нужен только для `/predict` без явной группы: сравнение
  уверенности двух CTC-моделей с непересекающимися алфавитами выбирает модель
  верно лишь ~75% раз, классификатор — ~97-100%.
* **Рост алфавита.** Если в новых данных встретились невиданные символы, выходной
  слой расширяется с сохранением старых весов — переобучать с нуля не нужно.

## Структура

```
src/
├── api/                  слой API: только эндпоинты
│   ├── app.py            сборка FastAPI и lifespan
│   ├── container.py      связывание слоёв (AppContainer)
│   ├── deps.py           зависимости эндпоинтов
│   └── routes/           health, status, predict, train
├── constants/            слой констант: все фиксированные значения
│   ├── groups.py         группы (words/digits) и режимы роутинга
│   ├── defaults.py       значения по умолчанию для настроек
│   ├── db.py             регулярка идентификаторов, префиксы DSN, пул
│   ├── storage.py        имена файлов, ключи чекпоинтов и state.json
│   ├── images.py         препроцессинг и параметры аугментации
│   ├── networks.py       архитектура CRNN и классификатора
│   ├── training.py       гиперпараметры обучения
│   ├── api.py            пути, теги, имена фоновых задач
│   ├── cli.py            имена команд и их значения по умолчанию
│   └── demo.py           параметры генератора синтетики
├── templates/            слой текстовых шаблонов: всё, что читает человек
│   ├── errors.py         тексты исключений и HTTP-ошибок
│   ├── logs.py           шаблоны логов (%-стиль, ленивое форматирование)
│   ├── notes.py          пояснения в результате обучения
│   ├── api.py            заголовки, summary эндпоинтов, подписи полей
│   ├── cli.py            справка и вывод команд
│   └── sql.py            DDL демо-таблицы
├── schemas/              слой схем: DTO для API и сервисов (pydantic)
│   ├── common.py         GroupName, RoutingMode
│   ├── images.py         ImageRow, IngestStats
│   ├── prediction.py     PredictRequest, Candidate, PredictionResult
│   ├── status.py         ServiceStatus, GroupStatus, ClassifierStatus, CursorState
│   └── training.py       TrainKind, TrainResult, GroupTrainingState, EvalResult
├── models/               слой моделей данных: ORM-описание таблиц (SQLAlchemy 2.0)
│   └── labeled_image.py  фабрика модели под имена таблицы и колонок из настроек
├── services/             слой сервисов приложения
│   ├── images.py         ImageRepository — чтение таблицы с картинками
│   ├── prediction.py     PredictionService — инференс и выбор модели
│   └── status.py         StatusService — сводное состояние
├── training/             слой непрерывного обучения
│   ├── networks.py       CRNN и CTC-декодирование
│   ├── charset.py        алфавит модели
│   ├── dataset.py        декодирование base64, препроцессинг, батчи
│   ├── store.py          локальный кэш датасета
│   ├── trainer.py        обучение и дообучение одной группы
│   ├── classifier.py     классификатор «слово или число»
│   ├── registry.py       ModelRegistry — живые модели в памяти
│   └── service.py        TrainingService — опрос базы и оркестрация обучения
├── config.py             настройки
├── demo.py               генератор синтетических данных
└── cli.py                команды

tests/
├── unit/                 константы, шаблоны, алфавит, настройки, кэш, датасет,
│                         сети, SQL, сервис обучения
├── api/                  эндпоинты через TestClient
└── integration/          полный цикл обучения (slow) и репозиторий на PostgreSQL
```

Зависимости направлены в одну сторону:
`api -> services -> training -> models/schemas -> constants/templates`.
Эндпоинты не знают ни про torch, ни про SQL; сервис обучения не знает про HTTP;
`constants/` и `templates/` не зависят ни от чего.

Числовых и строковых литералов вне `constants/` в коде нет, текста для человека —
вне `templates/`. Последнее проверяется тестом: он разбирает AST всех модулей и
падает, если находит русский текст в литерале за пределами слоя шаблонов.

## Установка

```bash
uv sync
cp .env.example .env   # и прописать CL_PG_DSN, CL_TABLE и имена колонок
```

## Запуск

```bash
# разовая выкачка всей таблицы в кэш + обучение обеих моделей с нуля
uv run continuous-learning bootstrap

# боевой режим: HTTP API + фоновое слежение за базой и дообучение
uv run continuous-learning run

# то же, но без HTTP
uv run continuous-learning watch

# состояние моделей
uv run continuous-learning status

# ручной запуск обучения
uv run continuous-learning train words --kind full
uv run continuous-learning train digits --kind incremental

# распознать картинку из файла
uv run continuous-learning predict ./sample.png
```

## Тесты

```bash
uv run pytest                  # всё, ~6 секунд
uv run pytest -m "not slow"    # без реального обучения моделей
```

Тесты на репозиторий PostgreSQL пропускаются, пока не задан DSN:

```bash
docker run -d --name cl-test -e POSTGRES_PASSWORD=test -p 55433:5432 postgres:17-alpine
CL_TEST_PG_DSN=postgresql://postgres:test@127.0.0.1:55433/postgres uv run pytest tests/integration
```

### Линтеры и форматер

```bash
uv run ruff check .          # правила
uv run ruff check . --fix    # автоисправление
uv run ruff format .         # форматирование
uv run mypy                  # типы
```

Настройки — в `pyproject.toml`. Ruff держит строку в 100 символов и включает
`E/W/F/I/UP/B/C4/SIM/RUF`; выключены только `RUF001-003` — они ругаются на
кириллицу в комментариях, считая «с» и «р» похожими на латинские. Импорты
сортирует сам ruff (`I`), все модули верхнего уровня перечислены как свои:
пакет разложен по `src/` без промежуточной директории.

Те же три проверки стоят на пре-коммите (`.pre-commit-config.yaml`). Хуки
локальные и запускаются через `uv`, поэтому версии берутся из `uv.lock` — на
коммите и в CI работает ровно одно и то же. Включить после `uv sync`:

```bash
uv run pre-commit install        # один раз на свежем клоне
uv run pre-commit run --all-files  # прогнать по всему репозиторию
git commit --no-verify           # разово пропустить хуки
```

`ruff check --fix` и `ruff format` правят файлы сами: если они что-то изменили,
коммит прерывается, изменения остаются в рабочем дереве — достаточно `git add`
и повторить коммит. Mypy ничего не правит и проверяет `src/` целиком, а не
только изменённые файлы: по одному файлу он не видит импортирующие модули.

Определения сетей в `training/networks.py` и `training/classifier.py` обёрнуты
в `# fmt: off`: слои там выписаны таблицей с комментариями про размер тензора,
а форматер разложил бы их по одному на строку и сломал выравнивание.

Mypy проверяет `src/` с `check_untyped_defs`. Тесты не проверяются: они
подставляют в `TrainingService` фейковый репозиторий вместо `ImageRepository`,
и чтобы это было корректно с точки зрения типов, зависимость пришлось бы
описать протоколом.

### Проверка без базы

```bash
uv run continuous-learning selftest --initial 300 --new 100 --epochs 25
```

Генерирует синтетические картинки (слова и числа), обучает обе модели, доливает
по 100 новых примеров, дообучает и печатает метрики до/после. На Apple Silicon
полный цикл занимает ~5 минут; чем больше `--epochs`, тем ниже итоговый CER.

### Демо-данные в Postgres

```bash
uv run continuous-learning seed-db --count 300
```

Создаёт таблицу из `.env` (если её нет) и заливает синтетические строки — удобно,
чтобы прогнать весь цикл, не трогая боевую базу.

## HTTP API

| Метод | Путь | Описание |
|---|---|---|
| `GET`  | `/health` | живость |
| `GET`  | `/status` | версии моделей, метрики, сколько строк ждёт дообучения |
| `POST` | `/predict` | `{"image": "<base64>", "group": "words"\|"digits"\|null}` |
| `POST` | `/train/{group}?kind=full\|incremental` | ручной запуск обучения |

Без `group` модель выбирает классификатор (`routed_by: classifier`,
`group_confidence`); пока он не обучен — берётся ответ с большей уверенностью
(`routed_by: ocr_confidence`). В `candidates` всегда возвращаются оба варианта:

```json
{
  "group": "words", "text": "город", "confidence": 0.9278,
  "routed_by": "classifier", "group_confidence": 0.8469,
  "candidates": [
    {"group": "words",  "text": "город",  "confidence": 0.9278},
    {"group": "digits", "text": "99994",  "confidence": 0.951}
  ]
}
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d "{\"image\": \"$(base64 -i sample.png)\"}" | jq
```

## Настройки (`.env`, префикс `CL_`)

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `CL_PG_DSN` | `postgresql://postgres:postgres@localhost:5432/postgres` | подключение к базе |
| `CL_TABLE` / `CL_ID_COLUMN` / `CL_IMAGE_COLUMN` / `CL_ANSWER_COLUMN` | `labeled_images` / `id` / `image` / `answer` | таблица и колонки |
| `CL_CURSOR_COLUMN` | не задана | колонка для keyset-обхода новых строк, например `created_at` |
| `CL_POLL_INTERVAL` | `5` | период опроса базы, секунды |
| `CL_RETRAIN_THRESHOLD` | `100` | сколько новых строк в группе запускают дообучение |
| `CL_MIN_SAMPLES_TO_TRAIN` | `64` | минимум примеров для первого обучения |
| `CL_INITIAL_EPOCHS` / `CL_INCREMENTAL_EPOCHS` | `80` / `10` | эпохи полного обучения и дообучения |
| `CL_LR` / `CL_INCREMENTAL_LR` | `3e-4` / `1e-4` | learning rate |
| `CL_REPLAY_RATIO` | `4.0` | старых примеров на один новый при дообучении |
| `CL_CER_REGRESSION_TOLERANCE` | `0.01` | допустимое ухудшение `val_cer` |
| `CL_DEVICE` | `auto` | `auto` / `mps` / `cuda` / `cpu` |
| `CL_IMG_HEIGHT` / `CL_MAX_IMG_WIDTH` | `32` / `384` | препроцессинг (высота кратна 32) |
| `CL_BATCH_SIZE` | `32` | размер батча |

## Состояние на диске

```
var/
├── data/<group>/images/<id>.png   кэш картинок
├── data/<group>/labels.jsonl      метки
├── checkpoints/<group>.pt         веса + алфавит + метрики
├── checkpoints/group_classifier.pt  роутер «слово или число»
└── state.json                     курсор, пропущенные id, очередь на дообучение
```

Сервис можно перезапускать: курсор, кэш и чекпоинты переживают рестарт, повторной
выкачки базы не происходит.

## Требования к таблице

* `id` — `uuid` (подойдёт и любой другой тип: в SQL он приводится к тексту);
* `image` — base64 (допускается `data:image/png;base64,...`) или сырые байты картинки;
* `answer` — текст с картинки.

* желательно `created_at timestamptz` + индекс `(created_at, id)` — тогда опрос
  базы идёт по keyset-курсору, а не сверкой всех id.

Строки с битой картинкой пропускаются: их id запоминаются в `state.json`
(`skipped_rows` в `/status`), повторно они не вычитываются. Туда же попадают
строки, отсеянные фильтром на приёме: пустой ответ и число короче
`MIN_DIGITS_ANSWER_LENGTH` (5) символов.

## CI/CD

```
.pre-commit-config.yaml   ruff check / ruff format / mypy на коммите
Dockerfile                образ: uv-сборка зависимостей, рантайм python:3.13-slim
k8s/                      kustomize-оверлей: namespace, pvc, deployment, service, ingress
.github/workflows/ci.yml  тесты, сборка образа и проверка манифестов
.github/workflows/cd.yml  публикация образа в GHCR и выкатка в k3s
```

### CI

Запускается на pull request и на push в `main`/`master`, четыре независимые задачи:

* **Линтеры и типы** — `ruff check`, `ruff format --check`, `mypy`.
* **Тесты** — `uv sync --frozen` и весь `pytest`, включая `slow` (обучение
  настоящих моделей, ~10 секунд) и `integration`: рядом поднимается
  `postgres:17-alpine`, его DSN уходит в `CL_TEST_PG_DSN`. Перед тестами
  приложение собирается через `create_app(run_watcher=False)` — так видно
  ошибки импорта и валидации настроек без соединения с базой.
* **Сборка образа** — `docker build` без публикации, чтобы сломанный Dockerfile
  был виден на pull request, а не на выкатке.
* **Манифесты k8s** — `kubectl kustomize k8s`.

### CD

Push в `main`/`master` (или запуск вручную) собирает образ на `linux/amd64`,
пушит в GHCR двумя тегами (`<sha>` и `latest`), затем задача `deploy`
(окружение `production`, можно повесить ручное подтверждение) поднимает
WireGuard, подкладывает kubeconfig, собирает секрет и раскатывает оверлей:

```
kustomize edit set image continuous-learning=<образ>:<sha>
kustomize build k8s | kubectl apply -f -
kubectl -n rtt rollout status deployment/continuous-learning
```

Секреты репозитория (окружение `production`):

| секрет | зачем |
| --- | --- |
| `WG_CONFIG` | конфиг WireGuard в base64 — через него раннер видит кластер |
| `KUBE_API` | `https://<нода>:6443`, проверка доступности перед выкаткой |
| `KUBE_CONFIG` | kubeconfig (base64 или открытым текстом), без `127.0.0.1` в `server` |
| `CL_PG_DSN` | строка подключения к Postgres, кладётся в секрет кластера `continuous-learning-db` под тем же именем |

Секрет создаётся на выкатке через `kubectl create secret --dry-run=client`,
в репозитории не хранится и удаляется с раннера в `always()`-шаге.

### Образ

Зависимости ставятся отдельным слоем (`uv sync --locked --no-install-project
--no-dev`), код копируется поверх. Под linux `torch` берётся из индекса
`pytorch-cpu` (`[tool.uv.sources]` в `pyproject.toml`) — колёса `nvidia-*`
весят несколько гигабайт, а в кластере обучение всё равно идёт на CPU; образ
получается ~650 МБ. На macOS `torch` по-прежнему ставится с PyPI, со
сборкой под MPS.

Процесс запускается как `uvicorn --factory api:create_app` от пользователя
`10001`, корень контейнера смонтирован только на чтение.

### Выкатка в кластер

Под один (`strategy: Recreate`): том `ReadWriteOnce`, а веса живут в памяти
процесса — вторая реплика писала бы в тот же кэш и те же чекпоинты.
`var/` вынесен на `PersistentVolumeClaim continuous-learning-var` (5 Gi,
local-path в k3s): без тома после рестарта сервис заново выкачает таблицу и
начнёт обучение с нуля.

Настройки таблицы и темпа обучения лежат в `env` деплоймента
(`CL_TABLE`, `CL_CURSOR_COLUMN`, `CL_POLL_INTERVAL`, `CL_RETRAIN_THRESHOLD`),
строка подключения — в секрете. `OMP_NUM_THREADS` выставлен по лимиту CPU:
иначе torch берёт потоки по числу ядер ноды и обучение толкается само с собой.

Ingress — `traefik`, хост `ocr.home`, TLS выключен (API без браузерных фич);
включается раскомментированием блока в `k8s/ingress.yaml`.

Руками, без GitHub Actions:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml   # прописать CL_PG_DSN
kubectl apply -f k8s/secret.yaml
kustomize build k8s | kubectl apply -f -
```
