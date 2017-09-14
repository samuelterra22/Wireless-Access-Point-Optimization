import ezdxf

#dwg = ezdxf.readfile("/home/samuel/Desktop/IFMG/9 Periodo/TCC/bloco-A.dxf")
dwg = ezdxf.readfile("../DXFs/bloco-A-l.dxf")
#dwg = ezdxf.readfile("/home/samuel/Desktop/IFMG/9 Periodo/TCC/bloco-A-c.dxf")
#dwg = ezdxf.readfile("/home/samuel/Desktop/IFMG/9 Periodo/TCC/retangulo1.dxf")

modelspace = dwg.modelspace()

saida_pygame = ''

escala = 25


xMin = -1
yMin = -1
for e in modelspace:
    if e.dxftype() == 'LINE' and e.dxf.layer == 'ARQ':
        if e.dxf.start[0] < xMin or xMin == -1:
            xMin = e.dxf.start[0]
        if e.dxf.start[1] < yMin or yMin == -1:
            yMin = e.dxf.start[1]


for e in modelspace:
    if e.dxftype() == 'LINE':
        if e.dxf.layer == 'ARQ':
            # print("LINE on layer: %s\n" % str(e.dxf.layer))
            # print("start point: %s\n" % str(e.dxf.start))
            # print("end point: %s\n" % str(e.dxf.end))
            #teste += 'draw.line((('+str(e.dxf.start[0]*tam)+', '+str(e.dxf.start[1]*tam)+'), ('+str(e.dxf.end[0]*tam)+', '+str(e.dxf.end[1]*tam)+')), fill=128, width=3)\n'
            saida_pygame += 'pygame.draw.line(DISPLAYSURF, BLUE, ('+str(int((e.dxf.start[0]-xMin)*escala))+', '+str(int((e.dxf.start[1]-yMin)*escala))+'), ('+str(int((e.dxf.end[0]-xMin)*escala))+', '+str(int((e.dxf.end[1]-yMin)*escala))+'))\n'

# for e in dwg.entities:
#     print("DXF Entity: %s\n" % e.dxftype())

# for layer in dwg.layers:
#     if layer.dxf.name != '0':
#         layer.off()  # switch all layers off except layer '0'
#         print(layer.dxf.name)

f = open('saida', 'w')
f.write(saida_pygame)
f.close()

