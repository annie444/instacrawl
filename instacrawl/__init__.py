"""Top-level package for instagram crawler."""
# instacrawl/__init__.py

__app_name__ = "instacrawl"
<<<<<<< Updated upstream
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "to-do id error",
}
||||||| Stash base
__version__ = "0.1.3"
=======
__version__ = "0.1.4"
>>>>>>> Stashed changes
