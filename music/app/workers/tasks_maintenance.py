from celery import shared_task
import os
import shutil


@shared_task(name="app.workers.tasks_maintenance.retry_failed_jobs_task")
def retry_failed_jobs_task():
    pass


@shared_task(name="app.workers.tasks_maintenance.cleanup_tmp_files_task")
def cleanup_tmp_files_task():
    tmp_dir = "/tmp/music_factory"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
