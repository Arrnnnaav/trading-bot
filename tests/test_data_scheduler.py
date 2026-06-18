"""Tests for data ingestion scheduler wiring in main.py."""

import pytest

pytest.importorskip("apscheduler")


def test_fii_dii_job_registered():
    from main import _build_data_scheduler

    scheduler = _build_data_scheduler()
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert "fii_dii_daily" in job_ids, f"Expected fii_dii_daily job, got: {job_ids}"


def test_news_job_registered():
    from main import _build_data_scheduler

    scheduler = _build_data_scheduler()
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert "news_live_30min" in job_ids, f"Expected news_live_30min job, got: {job_ids}"


def test_fii_dii_job_cron_trigger():
    from apscheduler.triggers.cron import CronTrigger
    from main import _build_data_scheduler

    scheduler = _build_data_scheduler()
    job = next(j for j in scheduler.get_jobs() if j.id == "fii_dii_daily")
    assert isinstance(job.trigger, CronTrigger)
    # Verify hour=19
    fields = {f.name: f for f in job.trigger.fields}
    assert str(fields["hour"]) == "19", f"Expected hour=19, got {fields['hour']}"
    # Verify day_of_week is mon-fri
    assert str(fields["day_of_week"]) in ("mon-fri", "0-4"), (
        f"Unexpected day_of_week: {fields['day_of_week']}"
    )


def test_news_job_cron_trigger():
    from apscheduler.triggers.cron import CronTrigger
    from main import _build_data_scheduler

    scheduler = _build_data_scheduler()
    job = next(j for j in scheduler.get_jobs() if j.id == "news_live_30min")
    assert isinstance(job.trigger, CronTrigger)
    fields = {f.name: f for f in job.trigger.fields}
    # Verify minute=0,30
    assert str(fields["minute"]) == "0,30", (
        f"Expected minute=0,30, got {fields['minute']}"
    )


def test_exactly_two_data_jobs():
    from main import _build_data_scheduler

    scheduler = _build_data_scheduler()
    assert len(scheduler.get_jobs()) == 2
