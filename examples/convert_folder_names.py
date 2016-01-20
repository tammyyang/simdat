# coding=utf-8

import os
import json

topdir = '.'
ext = ('jpg', 'png')

output = {}
counter = 1


def write(outputs, fname='./output.json'):
    """Output results to a json file

    @param fname: output filename

    """
    with open(fname, 'w') as f:
        json.dump(outputs, f)
    f.close()


for root, dirs, files in os.walk(topdir, topdown=False):
    for name in files:
        upper = os.path.basename(root)
        if isinstance(upper, str):
            upper = upper.decode('utf-8')
        else:
            upper = upper.encode('utf-8')
        if upper in output.keys():
            continue
        if not name.endswith(ext):
            continue
        folder_name = 'person-' + str(counter)
        output[upper] = folder_name
        new_root = os.path.dirname(root)
        ori_path = os.path.join(new_root, upper)
        new_path = os.path.join(new_root, folder_name)
        os.rename(ori_path, new_path)
        counter += 1

with open('./output.txt', 'w') as f:
    for p in output:
        f.write("%s: %s" % (p.encode('utf8'), output[p]))
        f.write('\n')

f.close()
