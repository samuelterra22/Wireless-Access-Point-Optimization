###############################################
## calculando a qtde de nros pares de 1 a 1001
import timeit

cmd = 'calculando a qtde de nros pares de 1 a 1001'
print(cmd)

cmd = """ 
sum([~x%2 for x in range(1,1001)])
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nsum([f() for...): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
count = 0
for x in range(1,1001): 
        count += ~x%2
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nfor...count+=f()): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
len(filter(lambda x: ~x%2, range(1,1001)) )
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nlen(filter(f())): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
sum(filter(lambda x: ~x%2, range(1,1001)) )
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nsum(filter(f())): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
sum(map(lambda x: ~x%2, range(1,1001)) )
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nsum(map(f())): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
reduce(lambda x,y: x+y, filter(lambda x: ~x%2, range(1,1001)) )
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nreduce(filter(f())): \n' + str(round(t1, 4) * 1000) + " ms")

cmd = """ 
reduce(lambda x,y: x+y, map(lambda x: ~x%2, range(1,1001)) )
"""
t1 = timeit.timeit(cmd, number=1000)
print('\nreduce(map(f())): \n' + str(round(t1, 4) * 1000) + " ms")
