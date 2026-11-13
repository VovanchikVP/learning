from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import uuid
from io import BytesIO
from pathlib import Path

from api.container import AppContainer
from config import Settings
from constants import cli as cc
from constants.groups import GROUPS
from schemas import TrainKind
from templates import cli as texts
from templates import sql
from utils import setup_logging


def metric(value: float | None, template: str) -> str:
    """Метрика может быть не измерена (пустая val-выборка) — тогда печатаем n/a."""
    return texts.METRIC_UNKNOWN if value is None else template.format(value=value)


def dump(model) -> str:
    return model.model_dump_json(indent=cc.JSON_INDENT)


def cmd_run(args: argparse.Namespace) -> int:
    """API + фоновый цикл непрерывного обучения."""
    import uvicorn

    from api import create_app

    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        log_level=args.log_level.lower(),
    )
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Только фоновый цикл, без HTTP."""
    container = AppContainer.build(Settings())

    async def main() -> None:
        try:
            await container.training.run()
        except asyncio.CancelledError:
            container.training.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        container.training.stop()
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Выкачать все строки из БД в локальный кэш и обучить обе модели с нуля."""

    async def main() -> int:
        container = AppContainer.build(Settings())
        training = container.training
        await training.bootstrap()
        await container.repository.close()
        for group in GROUPS:
            store = container.registry.store(group)
            if len(store) < container.settings.min_samples_to_train:
                print(texts.SKIP_NOT_ENOUGH.format(group=group, count=len(store)))
                continue
            if container.registry.trainer(group).is_ready and not args.force:
                print(texts.SKIP_ALREADY_TRAINED.format(group=group))
                continue
            result = await training.train_group(group, TrainKind.FULL)
            print(texts.TRAIN_RESULT.format(group=group, payload=dump(result)))
        return 0

    return asyncio.run(main())


def cmd_train(args: argparse.Namespace) -> int:
    async def main() -> int:
        container = AppContainer.build(Settings())
        result = await container.training.train_group(args.group, TrainKind(args.kind))
        print(dump(result))
        return 0 if result.accepted else 1

    return asyncio.run(main())


def cmd_status(args: argparse.Namespace) -> int:
    container = AppContainer.build(Settings())
    print(dump(container.status.status()))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from PIL import Image

    container = AppContainer.build(Settings())
    image = Image.open(args.path)
    image.load()
    print(dump(container.prediction.predict(image, args.group)))
    return 0


def cmd_seed_db(args: argparse.Namespace) -> int:
    """Создать демо-таблицу и залить в неё синтетические строки."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from demo import generate

    settings = Settings()

    async def main() -> int:
        engine = create_async_engine(settings.sqlalchemy_dsn)
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        sql.CREATE_DEMO_TABLE.format(
                            table=settings.table,
                            id_column=settings.id_column,
                            image_column=settings.image_column,
                            answer_column=settings.answer_column,
                        )
                    )
                )
                if settings.cursor_column:
                    await conn.execute(
                        text(
                            sql.CREATE_CURSOR_INDEX.format(
                                table=settings.table,
                                cursor_column=settings.cursor_column,
                                id_column=settings.id_column,
                            )
                        )
                    )
                rows: list[dict[str, str]] = []
                for group in GROUPS:
                    rows.extend(
                        {"image": image, "answer": answer}
                        for image, answer in generate(
                            group, args.count, seed=args.seed + len(group)
                        )
                    )
                await conn.execute(
                    text(
                        sql.INSERT_DEMO_ROW.format(
                            table=settings.table,
                            image_column=settings.image_column,
                            answer_column=settings.answer_column,
                        )
                    ),
                    rows,
                )
            print(texts.SEED_DONE.format(count=len(rows), per_group=args.count))
        finally:
            await engine.dispose()
        return 0

    return asyncio.run(main())


def cmd_selftest(args: argparse.Namespace) -> int:
    """Проверка пайплайна на синтетике, без обращения к БД."""
    from PIL import Image

    from demo import generate

    settings = Settings(
        workdir=Path(args.workdir),
        initial_epochs=args.epochs,
        incremental_epochs=max(
            cc.SELFTEST_MIN_INCREMENTAL_EPOCHS, args.epochs // cc.SELFTEST_INCREMENTAL_EPOCH_DIVISOR
        ),
        min_samples_to_train=cc.SELFTEST_MIN_SAMPLES,
        retrain_threshold=args.new,
        early_stop_patience=args.epochs,
    )
    container = AppContainer.build(settings)
    training = container.training

    def feed(count: int, seed: int) -> dict[str, int]:
        added = dict.fromkeys(GROUPS, 0)
        for group in GROUPS:
            for image_b64, answer in generate(group, count, seed=seed):
                row_id = str(uuid.uuid4())  # как в боевой таблице
                got = training.ingest_row(row_id, image_b64, answer)
                if got:
                    training.pending[got].append(row_id)
                    added[got] += 1
        training._save_state()
        return added

    async def main() -> int:
        print(texts.SELFTEST_STEP_GENERATE.format(count=args.initial))
        print(texts.SELFTEST_ADDED.format(added=feed(args.initial, seed=cc.SELFTEST_SEED_INITIAL)))
        for group in GROUPS:
            result = await training.train_group(group, TrainKind.FULL)
            print(
                texts.SELFTEST_FULL_RESULT.format(
                    group=group,
                    cer=metric(result.val_cer, texts.METRIC_CER),
                    accuracy=metric(result.val_accuracy, texts.METRIC_ACCURACY),
                )
            )

        print(texts.SELFTEST_STEP_INCREMENT.format(count=args.new))
        print(texts.SELFTEST_ADDED.format(added=feed(args.new, seed=cc.SELFTEST_SEED_NEW)))
        for group in GROUPS:
            result = await training.train_group(group, TrainKind.INCREMENTAL)
            print(
                texts.SELFTEST_INCREMENTAL_RESULT.format(
                    group=group,
                    new=result.new_samples,
                    baseline=metric(result.baseline_cer, texts.METRIC_CER),
                    cer=metric(result.val_cer, texts.METRIC_CER),
                    accuracy=metric(result.val_accuracy, texts.METRIC_ACCURACY),
                    accepted=result.accepted,
                    note=result.note,
                )
            )

        print(texts.SELFTEST_STEP_PREDICT)
        for group in GROUPS:
            image_b64, answer = generate(group, 1, seed=cc.SELFTEST_SEED_PREDICT)[0]
            image = Image.open(BytesIO(base64.b64decode(image_b64)))
            out = container.prediction.predict(image)
            print(
                texts.SELFTEST_PREDICTION.format(
                    group=group,
                    answer=answer,
                    predicted=out.group,
                    text=out.text,
                    routed_by=out.routed_by,
                )
            )
        return 0

    return asyncio.run(main())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=cc.PROG_NAME, description=texts.DESCRIPTION)
    parser.add_argument("--log-level", default=cc.DEFAULT_LOG_LEVEL)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(cc.COMMAND_RUN, help=texts.HELP_RUN)
    p.add_argument("--host"), p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(cc.COMMAND_WATCH, help=texts.HELP_WATCH)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser(cc.COMMAND_BOOTSTRAP, help=texts.HELP_BOOTSTRAP)
    p.add_argument("--force", action="store_true", help=texts.HELP_FORCE)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser(cc.COMMAND_TRAIN, help=texts.HELP_TRAIN)
    p.add_argument("group", choices=GROUPS)
    p.add_argument("--kind", choices=[k.value for k in TrainKind], default=TrainKind.FULL.value)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(cc.COMMAND_STATUS, help=texts.HELP_STATUS)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(cc.COMMAND_PREDICT, help=texts.HELP_PREDICT)
    p.add_argument("path")
    p.add_argument("--group", choices=GROUPS)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser(cc.COMMAND_SEED_DB, help=texts.HELP_SEED_DB)
    p.add_argument("--count", type=int, default=cc.DEFAULT_SEED_COUNT, help=texts.HELP_SEED_COUNT)
    p.add_argument("--seed", type=int, default=cc.DEFAULT_SEED_RANDOM_SEED)
    p.set_defaults(func=cmd_seed_db)

    p = sub.add_parser(cc.COMMAND_SELFTEST, help=texts.HELP_SELFTEST)
    p.add_argument("--workdir", default=cc.DEFAULT_SELFTEST_WORKDIR)
    p.add_argument("--initial", type=int, default=cc.DEFAULT_SELFTEST_INITIAL)
    p.add_argument("--new", type=int, default=cc.DEFAULT_SELFTEST_NEW)
    p.add_argument("--epochs", type=int, default=cc.DEFAULT_SELFTEST_EPOCHS)
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
