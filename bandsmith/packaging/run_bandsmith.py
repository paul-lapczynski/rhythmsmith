# PyInstaller's entry point (bandsmith.spec): the app as a script
import sys

from bandsmith.app import main

sys.exit(main())
