import re
import sys

inp = raw_input('Input field names, use space as delimiter!\n')

field_li = inp.split()

def suppress_field(f, field, only_str=False):
    if only_str:
        ptn = re.compile(field)
    else: 
        ptn = re.compile(field+r'\s*=')
    
    with open(f) as fi:
        line_li = fi.readlines()
    
    new_line_li = [l for l in line_li if ptn.search(l) is None]

    with open(f, 'w') as fi:
        fi.writelines(new_line_li)

for f in sys.argv[1:]:
    for field in field_li:
        suppress_field(f, field)
    
    suppress_field(f, 'WOS', True)
