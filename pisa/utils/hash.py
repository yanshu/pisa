import cPickle as pickle
import hashlib
import numpy as np
import struct

from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.profiler import line_profile, profile


__all__ = ['hash_obj', 'hash_file']


# NOTE: adding @line_profile decorator slows down function to order of 10s of
# ms even if set_verbosity(0)!

def hash_obj(obj, hash_to='int', full_hash=True):
    """Return hash for an object. Object can be a numpy ndarray or matrix
    (which is serialized to a string), an open file (which has its contents
    read), or any pickle-able Python object.

    Note that only the first most-significant 8 bytes (64 bits) from the MD5
    sum are used in the hash.

    Parameters
    ----------
    obj : object
        Object to hash. Note that the larger the object, the longer it takes to
        hash.

    hash_to : string
        'i', 'int', or 'integer': First 8 bytes of the MD5 sum are interpreted
            as an integer.
        'b', 'bin', or 'binary': MD5 sum digest; returns an 8-character string 
        'h', 'x', 'hex': MD5 sum hexdigest, (string of 16 characters)
        'b64', 'base64': first 8 bytes of MD5 sum are base64 encoded (with '+'
            and '-' as final two characters of encoding). Returns string of 11
            characters.

    full_hash : bool
        If True, hash on the full object's contents (which can be slow) or if
        False, hash on a partial object. For example, only a file's first kB is
        read, and only 1000 elements (chosen at random) of a numpy ndarray are
        hashed on. This mode of operation should suffice for e.g. a
        minimization run, but should _not_ be used for storing to/loading from
        disk.

    Returns
    -------
    hash_val : int or string

    See also
    --------
    hash_file : hash a file on disk by filename/path

    """
    FAST_HASH_FILESIZE_BYTES = int(1e4)
    FAST_HASH_NDARRAY_ELEMENTS = int(1e3)
    FAST_HASH_STR_BYTES = int(1e3)

    if hash_to is None:
        hash_to = 'int'
    hash_to = hash_to.lower()

    pass_on_kw = dict(hash_to=hash_to, full_hash=full_hash)

    # TODO: convert an existing hash to the desired type, if it isn't already
    # in this type
    if hasattr(obj, 'hash') and obj.hash is not None and obj.hash == obj.hash:
        return obj.hash

    # Handle numpy arrays and matrices specially
    if isinstance(obj, np.ndarray) or isinstance(obj, np.matrix):
        if full_hash:
            return hash_obj(obj.tostring(), **pass_on_kw)
        if isinstance(obj, np.matrix):
            obj = np.array(obj)
        flat = obj.ravel()
        len_flat = len(flat)
        stride = 1 + (len_flat // FAST_HASH_NDARRAY_ELEMENTS)
        sub_elements = flat[0::stride]
        return hash_obj(sub_elements.tostring(), **pass_on_kw)

    # Handle an open file object as a special case
    if isinstance(obj, file):
        if full_hash:
            return hash_obj(obj.read(), **pass_on_kw)
        return hash_obj(obj.read(FAST_HASH_FILESIZE_BYTES), **pass_on_kw)

    # Convert to string (if not one already) in a fast and generic way: pickle;
    # this creates a binary string, which is fine for sending to hashlib
    if not isinstance(obj, basestring):
        try:
            pkl = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        except:
            logging.error('Failed to pickle `obj` "%s" of type "%s"'
                          %(obj, type(obj)))
            raise
        obj = pkl

    if full_hash:
        md5hash = hashlib.md5(obj)
    else:
        # Grab just a subset of the string by changing the stride taken in the
        # character array (but if the string is less than
        # FAST_HASH_FILESIZE_BYTES, use a stride length of 1)
        stride = 1 + (len(obj) // FAST_HASH_STR_BYTES)
        md5hash = hashlib.md5(obj[0::stride])

    if hash_to in ['i', 'int', 'integer']:
        hash_val, = struct.unpack('<q', md5hash.digest()[:8])
    elif hash_to in ['b', 'bin', 'binary']:
        hash_val = md5hash.digest()[:8]
    elif hash_to in ['h', 'x', 'hex', 'hexadecimal']:
        hash_val = md5hash.hexdigest()[:16]
    elif hash_to in ['b64', 'base64']:
        hash_val = base64.b64encode(md5hash.digest()[:8], '+-')
    else:
        raise ValueError('Unrecognized `hash_to`: "%s"' % (hash_to,))
    return hash_val


def hash_file(fname, hash_to=None, full_hash=True):
    """Return a hash for a file, passing contents through hash_obj function."""
    resource = find_resource(fname)
    with open(resource, 'rb') as f:
        return hash_obj(f, hash_to=hash_to, full_hash=full_hash)


def test_hash_obj():
    assert hash_obj('x') == 3783177783470249117
    assert hash_obj('x', full_hash=False) == 3783177783470249117
    assert hash_obj('x', hash_to='hex') == '9dd4e461268c8034'
    assert hash_obj(object) == -591373952375362512
    assert hash_obj(object()) != hash_obj(object)

    for nel in [10, 100, 1000]:
        rs = np.random.RandomState(seed=0)
        a = rs.rand(nel, nel, 2)
        a0_h_full = hash_obj(a)
        a0_h_part = hash_obj(a, full_hash=False)

        rs = np.random.RandomState(seed=1)
        a = rs.rand(nel, nel, 2)
        a1_h_full = hash_obj(a)
        a1_h_part = hash_obj(a, full_hash=False)

        rs = np.random.RandomState(seed=2)
        a = rs.rand(nel, nel, 2)
        a2_h_full = hash_obj(a)
        a2_h_part = hash_obj(a, full_hash=False)

        assert a1_h_full != a0_h_full
        assert a2_h_full != a0_h_full
        assert a2_h_full != a1_h_full

        assert a1_h_part != a0_h_part
        assert a2_h_part != a0_h_part
        assert a2_h_part != a1_h_part

    logging.info('<< PASSED : test_hash_obj >>')

# TODO: test_hash_file function requires a "standard" file to test on
def test_hash_file():
    file_hash = hash_file('../utils/hash.py')
    logging.info(file_hash)
    file_hash = hash_file('../utils/hash.py', full_hash=False)
    logging.info(file_hash)
    logging.info('<< PASSED : test_hash_file >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_hash_obj()
    test_hash_file()
