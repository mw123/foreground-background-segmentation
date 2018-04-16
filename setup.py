import sys
import os, errno


this_dir = os.path.dirname(__file__)


def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)
        
def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)

mkdir_p(os.path.join(this_dir, 'weights'))
mkdir_p(os.path.join(this_dir, 'out'))

