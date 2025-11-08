
import sys
import asyncio
from src.launcher import ares_launcher

def run():
    sys.argv = [
        'src/launcher/ares_launcher.py',
        '--train-analyst-base',
        '--symbol',
        'ETHUSDT',
        '--execution-mode',
        'light'
    ]
    asyncio.run(ares_launcher.main())

if __name__ == '__main__':
    run()
