"""A group of functions to help Python 2 act like Python 3"""
import sys


def run_script(path):
    """Run a script from a file"""
    # In Python 2, we can use execfile(...), but in Python 3 that function
    # doesn't exist, and we instead use exec(open(...)).  Since the latter
    # approach always works, just use that.
    with open(path) as fid:
        exec(fid.read())


def print_stdout(msg):
    """Print to standard output"""
    # In Python 3, the write(...) function returns a value, so store that value
    # in a dummy variable so that it doesn't print.
    dummy = sys.stdout.write(msg + '\n')


def print_stderr(msg):
    """Print to standard error"""
    # In Python 3, the write(...) function returns a value, so store that value
    # in a dummy variable so that it doesn't print.
    dummy = sys.stderr.write(msg + '\n')


def print_msg(msg, output_channel='stdout'):
    """Print a string to standard output or standard error"""
    if output_channel.upper() == 'STDOUT':
        print_stdout(msg)
    elif output_channel.upper() == 'STDERR':
        print_stderr(msg)
    else:
        raise ValueError(
            'Invalid output channel.  Choose from the strings STDOUT, STDERR.')


# If running Python 2, make the range function act like xrange.  That is
# essentially what Python 3 does.
try:
    xrange
    range = xrange
# For Python 3, do nothing.
except NameError:
    pass
