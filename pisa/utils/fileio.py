# author: Justin Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   2015-06-13
"""Generic file I/O, dispatching specific file readers/writers as necessary"""


import cPickle
import os
import re

import pisa.utils.config_parser
from pisa.utils import hdf
from pisa.utils import jsons
from pisa.utils.log import logging
from pisa.utils import resources


JSON_EXTS = ['json']
HDF5_EXTS = ['hdf', 'h5', 'hdf5']
PKL_EXTS = ['pickle', 'pkl', 'p']
CFG_EXTS = ['ini', 'cfg']


def expandPath(path, exp_user=True, exp_vars=True, absolute=False):
    """Convenience function for expanding a path"""
    if exp_user:
        path = os.path.expanduser(path)
    if exp_vars:
        path = os.path.expandvars(path)
    if absolute:
        path = os.path.abspath(path)
    return path


def mkdir(d, mode=0750, warn=True):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    """
    try:
        os.makedirs(d, mode=mode)
    except OSError as err:
        if err[0] == 17:
            if warn:
                logging.warn('Directory "%s" already exists' %d)
        else:
            raise err
    else:
        logging.info('Created directory "%s"' %d)


NSORT_RE = re.compile("(\\d+)")
def nsort(l):
    """Numbers sorted by value, not by alpha order.

    Code from
    nedbatchelder.com/blog/200712/human_sorting.html#comments
    """
    return sorted(l, key=lambda a: zip(NSORT_RE.split(a)[0::2],
                                       map(int, NSORT_RE.split(a)[1::2])))


def findFiles(root, regex=None, fname=None, recurse=True, dir_sorter=nsort,
              file_sorter=nsort):
    """Find files by re or name recursively w/ ordering.

    Code adapted from
    stackoverflow.com/questions/18282370/python-os-walk-what-order

    Parameters
    ----------
    root : str
        Root directory at which to start searching for files
    regex : str or re.SRE_Pattern
        Only yield files matching `regex`.
    fname : str
        Only yield files matching `fname`
    recurse : bool
        Whether to search recursively down from the root directory
    dir_sorter
        Function that takes a list and returns a sorted version of it, for
        purposes of sorting directories
    file_sorter
        Function as specified for `dir_sorter` but used for sorting file names

    Yields
    ------
    fullfilepath : str
    basename : str
    match : re.SRE_Match or None
    """
    root = os.path.expandvars(os.path.expanduser(root))
    if isinstance(regex, basestring):
        regex = re.compile(regex)

    # Define a function for accepting a filename as a match
    if regex is None:
        if fname is None:
            def validfilefunc(fn):
                return True, None
        else:
            def validfilefunc(fn):
                if fn == fname:
                    return True, None
                return False, None
    else:
        def validfilefunc(fn):
            match = regex.match(fn)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for rootdir, dirs, files in os.walk(root):
            for basename in file_sorter(files):
                fullfilepath = os.path.join(root, basename)
                is_valid, match = validfilefunc(basename)
                if is_valid:
                    yield fullfilepath, basename, match
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(rootdir, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        is_valid, match = validfilefunc(basename)
                        if is_valid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(root)):
            fullfilepath = os.path.join(root, basename)
            #if os.path.isfile(fullfilepath):
            is_valid, match = validfilefunc(basename)
            if is_valid:
                yield fullfilepath, basename, match


def from_cfg(fname):
    config = pisa.utils.config_parser.BetterConfigParser()
    config.read(fname)
    return config


def from_pickle(fname):
    return cPickle.load(file(fname, 'rb'))


def to_pickle(obj, fname, overwrite=True):
    fpath = os.path.expandvars(os.path.expanduser(fname))
    if os.path.exists(fpath):
        if overwrite:
            logging.warn('Overwriting file at ' + fpath)
        else:
            raise Exception('Refusing to overwrite path ' + fpath)
    return cPickle.dump(obj, file(fname, 'wb'),
                        protocol=cPickle.HIGHEST_PROTOCOL)


def from_file(fname, fmt=None, **kwargs):
    """Dispatch correct file reader based on fmt (if specified) or guess
    based on file name's extension.

    Parameters
    ----------
    fname : string
        File path / name from which to load data.
    fmt : None or string
        If string, for interpretation of the file according to this format. If
        None, file format is deduced by an extension found in `fname`.
    **kwargs
        All other arguments are passed to the function called to read the file.

    Returns
    -------
    Object instantiated from the file (string, dictionariy, ...). Each format
    is interpreted differently.

    """
    if fmt is None:
        _, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    fname = resources.find_resource(fname)
    if ext in JSON_EXTS:
        return jsons.from_json(fname, **kwargs)
    if ext in HDF5_EXTS:
        return hdf.from_hdf(fname, **kwargs)
    if ext in PKL_EXTS:
        return from_pickle(fname, **kwargs)
    if ext in CFG_EXTS:
        return from_cfg(fname, **kwargs)
    errmsg = 'File "%s": unrecognized extension "%s"' % (fname, ext)
    logging.error(errmsg)
    raise TypeError(errmsg)


def to_file(obj, fname, fmt=None, **kwargs):
    """Dispatch correct file writer based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        _, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    if ext in JSON_EXTS:
        return jsons.to_json(obj, fname, **kwargs)
    elif ext in HDF5_EXTS:
        return hdf.to_hdf(obj, fname, **kwargs)
    elif ext in PKL_EXTS:
        return to_pickle(obj, fname, **kwargs)
    else:
        errmsg = 'Unrecognized file type/extension: ' + ext
        logging.error(errmsg)
        raise TypeError(errmsg)
