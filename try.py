f = open('write.txt', 'w')
f.write('1 2 3 4 5\n')
f.write('10 20 30 40 50\n')
f.close()


#read
a = []

f = open('write.txt', 'r')
for line in f:
    data = line.split()
    temp = []
    for number in data:
        temp.append(int(number))
        print temp
    a.append(temp)

print a
f.close()
