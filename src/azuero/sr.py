"""SlideRule Earth client setup.

The SlideRule client builds its own `requests.Session` with `trust_env=False`, which
ignores HTTPS_PROXY / CA-bundle environment variables. We create the global session with
`trust_env=True` so it works behind proxies too.
"""

from __future__ import annotations

import importlib
import logging

from sliderule.session import Session

from .config import load_config

_INIT = False


# `from sliderule import sliderule` returns the package (its __init__ rebinds the name),
# so fetch the submodule that owns the global session explicitly.
_srmod = importlib.import_module("sliderule.sliderule")


def init(timeout: tuple[int, int] = (30, 900)) -> Session:
    global _INIT
    if not _INIT:
        url = load_config()["icesat2"]["sliderule_url"]
        s = Session(domain=url, trust_env=True, rqst_timeout=timeout,
                    loglevel=logging.WARNING, rethrow=True)
        _srmod.slideruleSession = s
        # The stock version check sends a GET with a JSON body, which some proxies drop;
        # query the version without a body instead.
        info = s.source("version", None)
        logging.getLogger(__name__).info("SlideRule server %s", info["server"]["version"])
        _INIT = True
    return _srmod.slideruleSession
