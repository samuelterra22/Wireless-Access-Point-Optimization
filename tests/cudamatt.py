def maior_que_zero(x):
     return x > 0

valores = [10, 4, -1, 3, 5, -9, -11]
print (list(filter(maior_que_zero, valores)))