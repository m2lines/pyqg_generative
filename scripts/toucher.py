import os

for root, dirs, files in os.walk('/scratch/zanna'):
    print(root)
    os.system(f'touch -a -c {root}/*')