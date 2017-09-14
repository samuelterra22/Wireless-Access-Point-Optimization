matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

for line in matrix:
    for value in line:
        print(str(value), end=' ')
    print()

print()
print()


a = [0, 9]
b = [0, 0]

for x in range(len(matrix)):
    for y in range(len(matrix[0])):
        if a[0] <= x <= b[0] and a[1] <= y <= b[1]:
            matrix[x][y] = 1


for line in matrix:
    for value in line:
        print(str(value), end=' ')
    print()