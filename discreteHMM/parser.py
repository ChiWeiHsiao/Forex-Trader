import sys
from datetime import datetime
from datetime import timedelta

fp_in = open(sys.argv[1], 'r')
fp_out = open(sys.argv[2], 'w')

upper_time = datetime.strptime('20000101 00:00:03', '%Y%m%d %H:%M:%S')
now_max = -1
for line in fp_in:
    record = line.split(',')
    if record[0]!='EUR/USD':
        print('Data inconsistent')
        exit(1)

    now_time = datetime.strptime(record[1].split('.')[0], '%Y%m%d %H:%M:%S')
    if now_time > upper_time:
        if now_max > 0:
            print(now_max, file=fp_out)
        now_max = float(record[2])
        upper_time = now_time + timedelta(seconds=3)
    now_max = max(now_max, float(record[2]))

fp_in.close()
fp_out.close()