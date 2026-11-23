import os

import pytest

# Qt tests draw offscreen: no window on screen, no display needed. Set before any QApplication exists.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from air_strings.ui.main_window import ensure_application

    return ensure_application()
