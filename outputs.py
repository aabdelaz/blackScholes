import os, time

sizes = [200, 400, 800, 1600, 3200, 6400]
# sizes = [200]

f = open('times', 'w')

for n in sizes:
    f.write(str(n) + ' ');

    start = time.time()
    os.system('./cudaSolver ' + str(n));
    end = time.time();
    f.write(str(end - start) + ' ');
    
    start = time.time()
    os.system('./solver ' + str(n));
    end = time.time();
    f.write(str(end - start) + '\n');

f.close();
     
