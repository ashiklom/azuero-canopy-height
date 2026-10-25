"""NASA Earthdata authentication helpers.

Credentials are taken from EARTHDATA_USERNAME/EARTHDATA_PASSWORD (earthaccess
convention), falling back to EARTHDATA_LOGIN_USER/EARTHDATA_LOGIN_PASSWORD.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def login():
    import earthaccess

    for a, b in (("EARTHDATA_USERNAME", "EARTHDATA_LOGIN_USER"),
                 ("EARTHDATA_PASSWORD", "EARTHDATA_LOGIN_PASSWORD")):
        if not os.environ.get(a) and os.environ.get(b):
            os.environ[a] = os.environ[b]
    return earthaccess.login(strategy="environment")


def session():
    """`requests.Session` that follows Earthdata Login redirects."""
    import earthaccess

    login()
    return earthaccess.get_requests_https_session()


def gdal_env() -> dict:
    """GDAL config for authenticated /vsicurl/ reads of Earthdata Cloud COGs."""
    from .config import data_path

    login()
    netrc = data_path("auth", ".netrc")
    if not netrc.exists():
        netrc.write_text(f"machine urs.earthdata.nasa.gov login {os.environ['EARTHDATA_USERNAME']}"
                         f" password {os.environ['EARTHDATA_PASSWORD']}\n")
        netrc.chmod(0o600)
    return dict(GDAL_HTTP_NETRC="YES", GDAL_HTTP_NETRC_FILE=str(netrc),
                GDAL_HTTP_COOKIEFILE=str(data_path("auth", "cookies.txt")),
                GDAL_HTTP_COOKIEJAR=str(data_path("auth", "cookies.txt")),
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.h5,.nc",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES")
