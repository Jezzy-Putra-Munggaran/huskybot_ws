import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jezzy/huskybot_ws/install/huskybot_multicam_parallel'
