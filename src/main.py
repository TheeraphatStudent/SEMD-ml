import sys
import os
import shutil
from cli.main import main

if __name__ == '__main__':

    if not os.path.exists('.env') and os.path.exists('.env.example'):
        shutil.copy('.env.example', '.env')

    sys.exit(main())
