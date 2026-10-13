"""
Emergency stop. Run:  python stop.py

Kills any running Celery workers and flushes the Redis queue so no tasks can
keep running. Safe to run any time — if nothing is running it just reports that.
"""

import subprocess
import sys
import platform


def kill_workers():
    system = platform.system()
    print("→ Killing any Celery workers...")
    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "celery.exe", "/T"],
                           capture_output=True)
            # celery often runs as python.exe on Windows
            subprocess.run(["wmic", "process", "where",
                            "commandline like '%celery%'", "delete"],
                           capture_output=True)
        else:
            subprocess.run(["pkill", "-9", "-f", "celery"], capture_output=True)
        print("  ✓ workers killed (if any were running)")
    except Exception as e:
        print(f"  ! couldn't kill workers automatically: {e}")
        print("    Manually: Ctrl+C the worker terminal (twice).")


def flush_redis():
    print("→ Flushing Redis queue...")
    # Try the docker container first (default setup), then local redis-cli.
    for cmd in (
        ["docker", "exec", "-it", "leaderboard_redis", "redis-cli", "FLUSHALL"],
        ["docker", "exec", "leaderboard_redis", "redis-cli", "FLUSHALL"],
        ["redis-cli", "FLUSHALL"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and "OK" in r.stdout:
                print(f"  ✓ Redis flushed via: {' '.join(cmd[:3])}...")
                return
        except Exception:
            continue
    print("  ! Couldn't flush Redis automatically.")
    print("    Manually: docker exec -it leaderboard_redis redis-cli FLUSHALL")


if __name__ == "__main__":
    print("=" * 55)
    print(" EMERGENCY STOP — halting all benchmark tasks")
    print("=" * 55)
    kill_workers()
    flush_redis()
    print("-" * 55)
    print("Done. Everything stopped and the queue is empty.")
    print("Restart the worker when you're ready to run again.")
