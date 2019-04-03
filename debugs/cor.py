from colour import Color

hex = "#adf"

corR = int(Color(hex).get_red() * 255)
corG = int(Color(hex).get_green() * 255)
corB = int(Color(hex).get_blue() * 255)

cor = tuple([corR, corG, corB])

print(cor)
