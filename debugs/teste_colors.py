from colour import Color


def get_color_gradient(steps=256):
    return list(Color("green").range_to(Color("red"), steps))


cores = get_color_gradient()
cores.pop(0)
cores.pop(len(cores)-1)

cor = cores[0]

for cor in cores:
    print(cor)
