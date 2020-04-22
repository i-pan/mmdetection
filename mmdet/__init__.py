try:
    from .version import __version__, short_version
except ModuleNotFoundError:
    print('Unable to import from `version`')

__all__ = ['__version__', 'short_version']
