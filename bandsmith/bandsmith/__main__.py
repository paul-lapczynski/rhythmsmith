import sys

if len(sys.argv) > 1:
    from bandsmith.cli import main
else:
    from bandsmith.app import main

sys.exit(main())
