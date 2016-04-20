import re
import os
import sys

ptn_1 = re.compile(r'\\begin{document}\n')
ptn_2 = re.compile(r'\\end{document}\n')
ptn_3 = re.compile(r'\.tex')
ptn_4 = re.compile(r'trimmed')

def trim_file(file_name):
    file_handle = open(file_name)
    lines = file_handle.readlines()
    file_handle.close()

    num_line = len(lines)
    ra = range(num_line)
    for i in ra:
        if ptn_1.match(lines[i]):
            break
        lines[i] = '%' + lines[i]
    lines[i] = '%' + lines[i]

    ra.reverse()
    for i in ra:
        if ptn_2.match(lines[i]):
            break
        lines[i] = '%' + lines[i]
    lines[i] = '%' + lines[i]

    new_name = re.sub(ptn_3, r'_trimmed.tex', file_name)
    new_file = open(new_name, 'w')
    new_file.writelines(lines)
    new_file.close()

gen = os.walk(r'./')

if sys.argv[1] is 't':
    for path, subdirs, files in os.walk(r'./'):
        f_match = []
        for fi in files:
            if ptn_3.search(fi):
                f_match.append(path+r'/'+fi)
		print path+r'/'+fi
        for fi in f_match:
            trim_file(fi)
elif sys.argv[1] is 'f':
    trim_file(sys.argv[2])
else:
    for path, subdirs, files in os.walk(r'./'):
        for fi in files:
            if ptn_4.search(fi):
                file_full_path = path+r'/'+fi
                os.remove(file_full_path)
