#Observacion: Para que el codigo compile, se debe instalar pandas, matplotlib, tkinter y customtkinter
#Importar los módulos necesarios
import customtkinter as ctk
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import re
from matplotlib.patches import Ellipse, Arc
import random

# Definir las funciones necesarias

def fun(funcion, n1=False, n2=False, n3=False, n4=False):
    # Reemplazar las operaciones booleanas por operadores que Python reconoce (or, and, not)
    funcion = funcion.replace('v', " or ")
    funcion = funcion.replace('^', " and ")
    funcion = funcion.replace('x\'', 'not x').replace('y\'', 'not y').replace('z\'', 'not z').replace('w\'', 'not w')
    x = n1
    y = n2
    z = n3
    w = n4
    return eval(funcion)


def reorganizar_funcion(funcion_simplificada):
    # Convierte la función simplificada de SymPy a una cadena
    funcion_simplificada = str(funcion_simplificada)

    # Divide la función simplificada en términos individuales
    terminos = funcion_simplificada.split('|')

    # Inicializa listas para cada variable
    x_terminos = []
    y_terminos = []
    z_terminos = []
    w_terminos = []

    for termino in terminos:
        # Divide cada término en caracteres individuales
        caracteres = re.findall(r"[a-zA-Z'~]+", termino)

        # Inicializa variables para cada variable
        x = ''
        y = ''
        z = ''
        w = ''

        # Recorre los caracteres y asigna a las variables correspondientes
        for caracter in caracteres:
            if 'x' in caracter:
                x = caracter
            elif 'y' in caracter:
                y = caracter
            elif 'z' in caracter:
                z = caracter
            elif 'w' in caracter:
                w = caracter

        # Agrega los términos a las listas correspondientes
        x_terminos.append(x)
        y_terminos.append(y)
        z_terminos.append(z)
        w_terminos.append(w)

    # Reorganiza los términos en el orden deseado
    nuevos_terminos = []
    for i in range(len(terminos)):
        nuevo_termino = f"({x_terminos[i]}{y_terminos[i]}{z_terminos[i]}{w_terminos[i]})"
        nuevos_terminos.append(nuevo_termino)

    # Une los términos reorganizados en la nueva función simplificada
    funcion_reorganizada = ' | '.join(nuevos_terminos)

    return funcion_reorganizada

def FuncionSimplificada():
    funcion_usuario = boolEntry.get()

    # Reemplazar el formato del usuario al formato de Sympy
    funcion_sympy = funcion_usuario.replace('x\'', '~x').replace('y\'', '~y').replace('z\'', '~z').replace('w\'', '~w').replace('v', '|').replace('^', '&')
    #print(funcion_sympy)

    # Crear una expresión de Sympy a partir del formato de Sympy
    expression = sp.sympify(funcion_sympy)
    #print(expression)

    # Verificar si todos los valores de la tabla de verdad son True o False
    tabla_verdad = pd.DataFrame(np.zeros((2 ** numVarVar.get(),), dtype=int), columns=['f'])

    for i, row in tabla_verdad.iterrows():
        valores = [bool(int(bin(i)[2:].zfill(numVarVar.get())[j])) for j in range(numVarVar.get())]
        tabla_verdad.at[i, 'f'] = fun(funcion_usuario, *valores)

    if tabla_verdad['f'].all():
        return "1"
    elif not tabla_verdad['f'].any():
        return "0"

    # Si no todos los valores son 1 o 0, simplificar la expresión
    simplified_expression = sp.simplify_logic(expression, force=True, form='dnf')
    simplified_expression = reorganizar_funcion(simplified_expression)
    #print(simplified_expression)

    # Reemplazar el formato de Sympy al formato del usuario
    funcion_simplificada = str(simplified_expression).replace('~x', "x'").replace('~y', "y'").replace('~z', "z'").replace('~w', "w'").replace('&', '^').replace('|', 'v')

    # Mostrar la función booleana simplificada en el formato del usuario
    #print(funcion_simplificada)
    return funcion_simplificada


def FunKarnaugh():
    funcion = FuncionSimplificada()
    return funcion.replace('(', '').replace(')', '').replace(' ^ ', '').replace('v', '+')
def generateResults():
    funcion = boolEntry.get()
    cant = numVarVar.get()

    resultados = []  # en esta lista irán los resultados de la función booleana

    if cant == 2:
        variable1 = [False, False, True, True]
        variable2 = [False, True, False, True]
        tabla_k = pd.DataFrame(np.zeros((2, 2), dtype=int), index=[False, True], columns=[False, True])
        for i in range(len(variable1)):
            result = fun(funcion, variable1[i], variable2[i])
            resultados.append(result)

            if result == True:
                tabla_k.at[variable1[i], variable2[i]] = 1
        tabla_k = tabla_k.rename({False: "x'", True: 'x'}, axis='index')
        tabla_k = tabla_k.rename({False: "y'", True: 'y'}, axis='columns')
        dict_ = {
            'x': variable1,
            'y': variable2,
        }
    elif cant == 3:
        variable1 = [False, False, False, False, True, True, True, True]
        variable2 = [False, False, True, True, False, False, True, True]
        variable3 = [False, True, False, True, False, True, False, True]
        tabla_k = pd.DataFrame(np.zeros((2, 4)), dtype=int, index=[False, True],
                               columns=["FalseFalse", "FalseTrue", "TrueTrue", "TrueFalse"])
        for i in range(len(variable1)):
            result = fun(funcion, variable1[i], variable2[i], variable3[i])
            resultados.append(result)

            if result == True:
                column = f"{variable2[i]}{variable3[i]}"
                tabla_k.at[variable1[i], column] = 1
        tabla_k = tabla_k.rename({False: "x'", True: 'x'}, axis='index')
        tabla_k = tabla_k.rename({"FalseFalse": "y'z'", "FalseTrue": "y'z", "TrueTrue": "yz", "TrueFalse": "yz'"},
                                 axis='columns')
        dict_ = {
            'x': variable1,
            'y': variable2,
            'z': variable3,
        }
    elif cant == 4:
        variable1 = [False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True]
        variable2 = [False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True]
        variable3 = [False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True]
        variable4 = [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True]
        tabla_k = pd.DataFrame(np.zeros((4, 4)), dtype=int,
                               index=["FalseFalse", "FalseTrue", "TrueTrue", "TrueFalse"],
                               columns=["FalseFalse", "FalseTrue", "TrueTrue", "TrueFalse"])
        for i in range(len(variable1)):
            result = fun(funcion, variable1[i], variable2[i], variable3[i], variable4[i])
            resultados.append(result)

            if result == True:
                ax = f"{variable1[i]}{variable2[i]}"
                column = f"{variable3[i]}{variable4[i]}"
                tabla_k.at[ax, column] = 1
        tabla_k = tabla_k.rename({"FalseFalse": "x'y'", "FalseTrue": "x'y", "TrueTrue": "xy", "TrueFalse": "xy'"},
                                 axis='index')
        tabla_k = tabla_k.rename({"FalseFalse": "z'w'", "FalseTrue": "z'w", "TrueTrue": "zw", "TrueFalse": "zw'"},
                                 axis='columns')
        dict_ = {
            'x': variable1,
            'y': variable2,
            'z': variable3,
            'w': variable4,
        }

    dict_['f'] = resultados
    tabla = pd.DataFrame(dict_)
    tabla = tabla.replace({False: 0, True: 1})

    # Dibujar ambas tablas con Matplotlib
    #fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    plt.suptitle(f'Función: {funcion}', fontsize=16, color='red')

    # Dibujar la tabla de verdad
    table_data = []
    headers = tabla.columns.tolist()
    table_data.append(headers)

    for row in tabla.itertuples(index=False):
        table_data.append(list(row))

    table = ax[0].table(cellText=table_data, loc='center', cellLoc='center', colLabels=None)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.5)
    ax[0].axis('off')
    ax[0].set_title('Tabla de Verdad', fontsize=14, color='blue')

    # Dibujar la tabla Karnaugh
    table_data = []
    headers = tabla_k.columns.tolist()
    table_data.append([""] + headers)
    valores_k = []

    for row in tabla_k.itertuples():
        row_data = [row[0]] + list(row[1:])
        table_data.append(row_data)
        valores_k.extend(list(row[1:]))

    table = ax[1].table(cellText=table_data, loc='center', cellLoc='center', colLabels=None)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.5)
    ax[1].axis('off')
    ax[1].set_title('Tabla Karnaugh', fontsize=14, color='blue')
    ax[1].text(0.5, 0.9, f'Funcion simplificada:', fontsize=12, ha='center', va='center', color='green')
    ax[1].text(0.4, 0.8, f'{FuncionSimplificada()}', fontsize=12, ha='center', va='center', color='green')
    ax[1].text(0.5, 0.3, f'Grupos:', fontsize=12, ha='center', va='center', color='green')
    ax[1].text(0.4, 0.2, f'{Agrupaciones_Karnaugh()}', fontsize=12, ha='center', va='center', color='green')

    # Subplot para agrupacionesGraficas()
    #ax[2] = plt.subplot(1, 3, 3)
    #ax[2].set_title('Agrupaciones Gráficas', fontsize=14, color='blue')
    agrupacionesGraficas()

    plt.tight_layout()
    plt.show()

    #agrupacionesGraficas()
    #print(valores_k)

def Agrupaciones_Karnaugh():
    n = numVarVar.get()
    cantidad_pos = list(range(1, 2 ** n + 1))
    func_simpli = FunKarnaugh()
    # Definir los valores para la tabla

    if n == 2:
        valores = [
            ["x'y'", "x'y"],
            ["xy'", "xy"]
        ]
    elif n == 3:
        valores = [
            ["x'y'z'", "x'y'z", "x'yz", "x'yz'"],
            ["xy'z'", "xy'z", "xyz", "xyz'"]
        ]
    elif n == 4:
        valores = [
            ["x'y'z'w'", "x'y'z'w", "x'y'zw", "x'y'zw'"],
            ["x'yz'w'", "x'yz'w", "x'yzw", "x'yzw'"],
            ["xyz'w'", "xyz'w", "xyzw", "xyzw'"],
            ["xy'z'w'", "xy'z'w", "xy'zw", "xy'zw'"]

        ]

    # Crear un DataFrame de Pandas a partir de los valores
    if n == 2:
        tabla = pd.DataFrame(valores, index=["Fila 1", "Fila 2"], columns=["Columna 1", "Columna 2"])
    elif n == 3:
        tabla = pd.DataFrame(valores, index=["Fila 1", "Fila 2"],
                             columns=["Columna 1", "Columna 2", "Columna 3", "Columna 4"])
    elif n == 4:
        tabla = pd.DataFrame(valores, index=["Fila 1", "Fila 2", "Fila 3", "Fila 4"],
                             columns=["Columna 1", "Columna 2", "Columna 3", "Columna 4"])

    # Mostrar la tabla original
    #print("Tabla original:")
    #print(tabla)

    if func_simpli == '1':
        # Todos los valores son 1, llenar la agrupación
        agrupacion_llenada = [[int(i) for i in cantidad_pos]]
        print("Grupos numéricos convertidos:", agrupacion_llenada)
        return agrupacion_llenada
    elif func_simpli == '0':
        agrupacion_llenada = None
        print("Grupos numéricos convertidos:", agrupacion_llenada)
        #print("Grupos numéricos convertidos: 0")
        return agrupacion_llenada

    else:
        terminos = re.split(r'\s*\+\s*', func_simpli)

        grupos_con_dos_letras = []
        grupos_con_tres_letras = []
        grupos_con_cuatro_letras = []
        grupo_general = []

        # Inicializar la lista de grupos numéricos
        grupos_numericos = []
        grupos_numericos_convertidos = []

        # Iterar a través de los grupos
        for grupo in terminos:
            # Verificar la cantidad de letras del grupo (sin tener en cuenta las comillas simples)
            if len(re.sub("'", "", grupo)) == 2:
                grupos_con_dos_letras.append(grupo)
            if len(re.sub("'", "", grupo)) == 3:
                grupos_con_tres_letras.append(grupo)
            if len(re.sub("'", "", grupo)) == 4:
                grupos_con_cuatro_letras.append(grupo)

        if len(grupos_con_dos_letras) > 0 and n == 2:
            grupo_general = grupos_con_dos_letras
        elif len(grupos_con_tres_letras) > 0 and n == 3:
            grupo_general = grupos_con_tres_letras
        elif len(grupos_con_cuatro_letras) > 0 and n == 4:
            grupo_general = grupos_con_cuatro_letras

        '''
        print(grupos_con_dos_letras)
        print(grupos_con_tres_letras)
        print(grupos_con_cuatro_letras)
        print(grupo_general)
        '''

        # Crear grupos numéricos basados en la posición de las celdas en la tabla
        for grupo in grupo_general:
            for index, row in tabla.iterrows():
                for col_name in tabla.columns:
                    if row[col_name] == grupo:
                        row_index, col_index = tabla.index.get_loc(index), tabla.columns.get_loc(col_name)
                        grupo_numerico = row_index * len(tabla.columns) + (col_index + 1)
                        grupos_numericos_convertidos.append(grupo_numerico)

        # Función para dividir los caracteres de los grupos en 'divi'
        def dividir_termino(termino):
            divi = re.findall(r"[a-zA-Z]'?|[a-zA-Z]+", termino)
            return divi

        # Recorrer cada término y compararlo con los elementos de la tabla
        for i, termino in enumerate(terminos, start=1):
            if termino not in grupo_general:
                termino_dividido = dividir_termino(termino)
                encontrados = []
                for index, row in tabla.iterrows():
                    for col_name in tabla.columns:
                        elementos_tabla = dividir_termino(row[col_name])
                        if all(item in elementos_tabla for item in termino_dividido):
                            encontrados.append((index, col_name))
                if encontrados:
                    if len(encontrados) > 1:
                        # Si el término se encuentra en más de un elemento de la tabla, agregar las posiciones
                        posiciones = [(fila, columna) for fila, columna in encontrados]
                        grupos_numericos.append(posiciones)
                    else:
                        # Si el término se encuentra en un solo elemento de la tabla, agregar solo esa posición
                        fila, columna = encontrados[0]
                        grupos_numericos.append((fila, columna))
                else:
                    grupos_numericos.append(
                        None)  # Agregar None si el término no se encuentra en ningún elemento de la tabla

        # Función para convertir la posición fila-columna a un valor numérico de 1 a 8
        def convertir_a_valor_numerico(fila, columna):
            filas = tabla.index.tolist()
            columnas = tabla.columns.tolist()
            fila_numerica = filas.index(fila) + 1
            columna_numerica = columnas.index(columna) + 1
            return (fila_numerica - 1) * len(columnas) + columna_numerica

        for grupo in grupos_numericos:
            if grupo is not None:
                posiciones_convertidas = [convertir_a_valor_numerico(fila, columna) for fila, columna in grupo]
                grupos_numericos_convertidos.append(posiciones_convertidas)
            else:
                grupos_numericos_convertidos.append(None)

        # grupos_numericos_convertidos.extend([grupos_con_tres_letras])

        # Mostrar la lista de grupos numéricos convertidos
        #print("Grupos numéricos convertidos:", grupos_numericos_convertidos)
        return grupos_numericos_convertidos

#-----------------------------------------------------------------------------------------

def agrupacionesGraficas():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Arc
    import random
    num_var = numVarVar.get()
    grupos = Agrupaciones_Karnaugh()
    if num_var == 2:
        # Datos para la tabla
        datos = [["y'", 'y'],
                 [1, 2],
                 [3, 4]]

        # Coordenadas de los puntos
        coordenadas = [(1, 1), (2, 1),
                       (1, 2), (2, 2)]
        # Crear una figura y un eje
        fig, ax = plt.subplots()

        texto_coordenadas = [(0, 2, "x'"),
                             (0, 1, "x")]

        # Agregar el texto a las coordenadas especificadas
        for x, y, texto in texto_coordenadas:
            ax.text(x, y, texto, fontsize=12, ha='center', va='center')  # Añadir texto al gráfico

        def conversion_grupos(grupos):
            coordenadas = {
                '1': (1, 2), '2': (2, 2),
                '3': (1, 1), '4': (2, 1)
            }

            lista_de_grupos = []  # Inicializamos una lista para almacenar los grupos

            for grupo in grupos:
                coordenadas_grupo = []  # Inicializamos una lista para las coordenadas del grupo
                for numero in grupo:
                    coordenada = coordenadas.get(str(numero))
                    if coordenada:
                        coordenadas_grupo.append(coordenada)
                lista_de_grupos.append(coordenadas_grupo)  # Agregamos las coordenadas del grupo a la lista de grupos

            return lista_de_grupos  # Devolvemos la lista de grupos

        resultado = conversion_grupos(grupos)

        def graficar_elipse(coord, colores_usados):
            n, m = 0, 0
            if (len(coord) == 1):
                n = 0.5
                m = 0.5
            if (len(coord) == 2):
                n = 0.7
                m = 0.7
            if (len(coord) == 4):
                n = 1
                m = 1
            # Calcular la media de las coordenadas x e y
            media_x = sum(x for x, _ in coord) / len(coord)
            media_y = sum(y for _, y in coord) / len(coord)

            # Calcular el ancho (width) como la diferencia entre el máximo y el mínimo valor de x
            max_x = max(x for x, _ in coord)
            min_x = min(x for x, _ in coord)
            width = max_x - min_x + n

            # Calcular la altura (height) como la diferencia entre el máximo y el mínimo valor de y
            max_y = max(y for _, y in coord)
            min_y = min(y for _, y in coord)
            height = max_y - min_y + m

            # Generar un color de borde aleatorio que no se haya usado previamente
            edgecolor = None
            while edgecolor is None or edgecolor in colores_usados:
                edgecolor = (random.random(), random.random(), random.random())

            colores_usados.add(edgecolor)  # Agregar el color utilizado a la lista de colores usados

            # Crear y dibujar la elipse con el borde de color aleatorio
            ellipse = Ellipse(xy=(media_x, media_y), width=width, height=height, edgecolor=edgecolor, fc='none', lw=2)
            ax.add_patch(ellipse)

        # Llamar a la función para dibujar la elipse para cada grupo en resultado
        colores_usados = set()  # Inicializar un conjunto para mantener un seguimiento de los colores utilizados
        for grupo in resultado:
            graficar_elipse(grupo, colores_usados)
            print(grupo)

        # Crear la tabla
        tabla = ax.table(cellText=datos, loc='center', cellLoc='center')

        # Estilo de la tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(12)
        tabla.scale(0.8, 5)  # Ajusta el tamaño de la tabla

        # Configurar los límites del eje
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 4)

        # Configurar el formato de los ticks en el eje x para mostrar enteros
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

        # Mostrar la figura
        plt.gca().set_aspect('equal', adjustable='box')  # Para mantener la relación de aspecto igual
        ax.axis('off')

    if num_var == 3:
        # Datos para la tabla
        datos = [["y'z'", "y'z", "yz", "yz'"],
                 [1, 2, 3, 4],
                 [5, 6, 7, 8]]

        # Coordenadas de los puntos
        coordenadas = [(1, 1), (2, 1), (3, 1), (4, 1),
                       (1, 2), (2, 2), (3, 2), (4, 2)]

        # Crear una figura y un eje
        fig, ax = plt.subplots()

        texto_coordenadas = [(0, 2, "x'"),
                             (0, 1, "x")]

        # Agregar el texto a las coordenadas especificadas
        for x, y, texto in texto_coordenadas:
            ax.text(x, y, texto, fontsize=12, ha='center', va='center')  # Añadir texto al gráfico

        def conversion_grupos(grupos):
            coordenadas = {
                '1': (1, 2), '2': (2, 2), '3': (3, 2), '4': (4, 2),
                '5': (1, 1), '6': (2, 1), '7': (3, 1), '8': (4, 1)
            }

            lista_de_grupos = []  # Inicializamos una lista para almacenar los grupos

            for grupo in grupos:
                coordenadas_grupo = []  # Inicializamos una lista para las coordenadas del grupo
                for numero in grupo:
                    coordenada = coordenadas.get(str(numero))
                    if coordenada:
                        coordenadas_grupo.append(coordenada)
                lista_de_grupos.append(coordenadas_grupo)  # Agregamos las coordenadas del grupo a la lista de grupos

            return lista_de_grupos  # Devolvemos la lista de grupos

        resultado = conversion_grupos(grupos)

        def graficar_elipse(coord, colores_usados):
            n, m = 0, 0
            if (len(coord) == 1):
                n = 0.3
                m = 0.3
            if (len(coord) == 2):
                n = 0.5
                m = 0.5
            if (len(coord) == 4):
                n = 0.7
                m = 0.7
            if (len(coord) == 8):
                n = 1
                m = 1
            # Calcular la media de las coordenadas x e y
            media_x = sum(x for x, _ in coord) / len(coord)
            media_y = sum(y for _, y in coord) / len(coord)

            # Calcular el ancho (width) como la diferencia entre el máximo y el mínimo valor de x
            max_x = max(x for x, _ in coord)
            min_x = min(x for x, _ in coord)
            width = max_x - min_x + n

            # Calcular la altura (height) como la diferencia entre el máximo y el mínimo valor de y
            max_y = max(y for _, y in coord)
            min_y = min(y for _, y in coord)
            height = max_y - min_y + m

            # Generar un color de borde aleatorio que no se haya usado previamente
            edgecolor = None
            while edgecolor is None or edgecolor in colores_usados:
                edgecolor = (random.random(), random.random(), random.random())

            colores_usados.add(edgecolor)  # Agregar el color utilizado a la lista de colores usados

            medias_x = []
            medias_y = []

            if min_x == 1 and max_x == 4 and 2 not in [x for x, _ in coord]:
                # Dividir el grupo en dos subgrupos
                subgrupo1 = [p for p in coord if p[0] == 1]
                subgrupo2 = [p for p in coord if p[0] == 4]

                # Calcular la media de las coordenadas x e y para el primer subgrupo
                media_x1 = sum(x for x, _ in subgrupo1) / len(subgrupo1)
                media_y1 = sum(y for _, y in subgrupo1) / len(subgrupo1)

                # Calcular la media de las coordenadas x e y para el segundo subgrupo
                media_x2 = sum(x for x, _ in subgrupo2) / len(subgrupo2)
                media_y2 = sum(y for _, y in subgrupo2) / len(subgrupo2)

                # esto aplica para ambos subgrupos
                max_x = max(x for x, _ in subgrupo2)
                min_x = min(x for x, _ in subgrupo2)
                width1 = max_x - min_x

                if (len(subgrupo1) == 1):
                    height = 1.8
                    width1 = -0.8

                # Dibujar un arco para el primer subgrupo   - x no varía en cada subgrupo
                arc1 = Arc((media_x1, media_y1), width1 + 1.5, height - 1, angle=270, theta1=0, theta2=180,
                           edgecolor=edgecolor, lw=2)
                ax.add_patch(arc1)

                # Dibujar un arco para el segundo subgrupo
                arc2 = Arc((media_x2, media_y2), width1 + 1.5, height - 1, angle=90, theta1=0, theta2=180,
                           edgecolor=edgecolor, lw=2)
                ax.add_patch(arc2)

                # Agregar las medias de x e y de ambos subgrupos a las listas
                medias_x.extend([media_x1, media_x2])
                medias_y.extend([media_y1, media_y2])
            else:
                # Crear y dibujar la elipse con el borde de color aleatorio
                ellipse = Ellipse(xy=(media_x, media_y), width=width, height=height, edgecolor=edgecolor, fc='none',
                                  lw=2)
                ax.add_patch(ellipse)

        # Llamar a la función para dibujar la elipse para cada grupo en resultado
        colores_usados = set()  # Inicializar un conjunto para mantener un seguimiento de los colores utilizados
        for grupo in resultado:
            graficar_elipse(grupo, colores_usados)
            print(grupo)

        # Crear la tabla
        tabla = ax.table(cellText=datos, loc='center', cellLoc='center')

        # Estilo de la tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(12)
        tabla.scale(0.8, 5)  # Ajusta el tamaño de la tabla

        # Configurar los límites del eje
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)

        # Configurar el formato de los ticks en el eje x para mostrar enteros
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

        # Mostrar la figura
        plt.gca().set_aspect('equal', adjustable='box')  # Para mantener la relación de aspecto igual
        ax.axis('off')

    if num_var == 4:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Arc
        import random

        # Datos para la tabla
        datos = [["z'w'", "z'w", "zw", "zw'"],
                 [1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]

        # Coordenadas de los puntos
        coordenadas = [(1, 1), (2, 1), (3, 1), (4, 1),
                       (1, 2), (2, 2), (3, 2), (4, 2),
                       (1, 3), (2, 3), (3, 3), (4, 3),
                       (1, 4), (2, 4), (3, 4), (4, 4)]

        # Crear una figura y un eje
        fig, ax = plt.subplots()

        texto_coordenadas = [(0, 4, "x'y'"),
                             (0, 3, "x'y"),
                             (0, 2, "xy"),
                             (0, 1, "xy'")]

        # Agregar el texto a las coordenadas especificadas
        for x, y, texto in texto_coordenadas:
            ax.text(x, y, texto, fontsize=12, ha='center', va='center')  # Añadir texto al gráfico

        def conversion_grupos(grupos):
            if grupos != None:
                coordenadas = {
                    '1': (1, 4), '2': (2, 4), '3': (3, 4), '4': (4, 4),
                    '5': (1, 3), '6': (2, 3), '7': (3, 3), '8': (4, 3),
                    '9': (1, 2), '10': (2, 2), '11': (3, 2), '12': (4, 2),
                    '13': (1, 1), '14': (2, 1), '15': (3, 1), '16': (4, 1)
                }

                lista_de_grupos = []  # Inicializamos una lista para almacenar los grupos

                for grupo in grupos:
                    coordenadas_grupo = []  # Inicializamos una lista para las coordenadas del grupo
                    for numero in grupo:
                        coordenada = coordenadas.get(str(numero))
                        if coordenada:
                            coordenadas_grupo.append(coordenada)
                    lista_de_grupos.append(
                        coordenadas_grupo)  # Agregamos las coordenadas del grupo a la lista de grupos

                return lista_de_grupos  # Devolvemos la lista de grupos
            else:
                return None

        resultado = conversion_grupos(grupos)

        def graficar_elipse(coord, colores_usados):
            n, m = 0, 0
            if coord != None:
                if (len(coord) == 1):
                    n = 0.5
                    m = 0.5
                if (len(coord) == 2):
                    n = 0.5
                    m = 0.5
                if (len(coord) == 4):
                    n = 0.7
                    m = 0.7
                if (len(coord) == 8):
                    n = 1
                    m = 1
                if (len(coord) == 16):
                    n = 1.6
                    m = 1.1
                # Calcular la media de las coordenadas x e y
                media_x = sum(x for x, _ in coord) / len(coord)
                media_y = sum(y for _, y in coord) / len(coord)

                # Calcular el ancho (width) como la diferencia entre el máximo y el mínimo valor de x
                max_x = max(x for x, _ in coord)
                min_x = min(x for x, _ in coord)
                width = max_x - min_x + n

                # Calcular la altura (height) como la diferencia entre el máximo y el mínimo valor de y
                max_y = max(y for _, y in coord)
                min_y = min(y for _, y in coord)
                height = max_y - min_y + m

                # Generar un color de borde aleatorio que no se haya usado previamente
                edgecolor = None
                while edgecolor is None or edgecolor in colores_usados:
                    edgecolor = (random.random(), random.random(), random.random())

                colores_usados.add(edgecolor)  # Agregar el color utilizado a la lista de colores usados
                medias_x = []
                medias_y = []

                if coord == [(1, 4), (4, 4), (1, 1), (4, 1)]:
                    # Dividir el grupo en cuatro subgrupos
                    subgrupo1 = [(1, 4)]
                    subgrupo2 = [(4, 4)]
                    subgrupo3 = [(1, 1)]
                    subgrupo4 = [(4, 1)]

                    # Calcular la media de las coordenadas x e y para el primer subgrupo
                    media_x1 = sum(x for x, _ in subgrupo1) / len(subgrupo1)
                    media_y1 = sum(y for _, y in subgrupo1) / len(subgrupo1)

                    # Calcular la media de las coordenadas x e y para el segundo subgrupo
                    media_x2 = sum(x for x, _ in subgrupo2) / len(subgrupo2)
                    media_y2 = sum(y for _, y in subgrupo2) / len(subgrupo2)

                    # Calcular la media de las coordenadas x e y para el primer subgrupo
                    media_x3 = sum(x for x, _ in subgrupo3) / len(subgrupo3)
                    media_y3 = sum(y for _, y in subgrupo3) / len(subgrupo3)

                    # Calcular la media de las coordenadas x e y para el segundo subgrupo
                    media_x4 = sum(x for x, _ in subgrupo4) / len(subgrupo4)
                    media_y4 = sum(y for _, y in subgrupo4) / len(subgrupo4)

                    arc1 = Arc((media_x1, media_y1), 0.5, 0.5, angle=90, theta1=90, theta2=270, edgecolor=edgecolor,
                               lw=2)
                    ax.add_patch(arc1)
                    arc2 = Arc((media_x2, media_y2), 0.5, 0.5, angle=90, theta1=90, theta2=270, edgecolor=edgecolor,
                               lw=2)
                    ax.add_patch(arc2)
                    arc3 = Arc((media_x3, media_y3), 0.5, 0.5, angle=270, theta1=90, theta2=270, edgecolor=edgecolor,
                               lw=2)
                    ax.add_patch(arc3)
                    arc4 = Arc((media_x4, media_y4), 0.5, 0.5, angle=270, theta1=90, theta2=270, edgecolor=edgecolor,
                               lw=2)
                    ax.add_patch(arc4)
                elif min_x == 1 and max_x == 4 and 2 not in [x for x, _ in coord]:
                    # Dividir el grupo en dos subgrupos
                    subgrupo1 = [p for p in coord if p[0] == 1]
                    subgrupo2 = [p for p in coord if p[0] == 4]
                    if (len(subgrupo2) == 2):
                        a = 1.5
                        b = 1
                    if (len(subgrupo2) == 4):
                        a = 4
                        b = 3
                    # Calcular la media de las coordenadas x e y para el primer subgrupo
                    media_x1 = sum(x for x, _ in subgrupo1) / len(subgrupo1)
                    media_y1 = sum(y for _, y in subgrupo1) / len(subgrupo1)

                    # Calcular la media de las coordenadas x e y para el segundo subgrupo
                    media_x2 = sum(x for x, _ in subgrupo2) / len(subgrupo2)
                    media_y2 = sum(y for _, y in subgrupo2) / len(subgrupo2)

                    # esto aplica para ambos subgrupos
                    max_x = max(x for x, _ in subgrupo2)
                    min_x = min(x for x, _ in subgrupo2)
                    width1 = max_x - min_x

                    if (len(subgrupo1) == 1):
                        height = 3
                        width1 = 3
                        a = -2.2
                        b = 2

                    # Dibujar un arco para el primer subgrupo   - x no varía en cada subgrupo
                    arc1 = Arc((media_x1, media_y1), width1 + a, height - b, angle=270, theta1=0, theta2=180,
                               edgecolor=edgecolor, lw=2)
                    ax.add_patch(arc1)

                    # Dibujar un arco para el segundo subgrupo
                    arc2 = Arc((media_x2, media_y2), width1 + a, height - b, angle=90, theta1=0, theta2=180,
                               edgecolor=edgecolor, lw=2)
                    ax.add_patch(arc2)

                    # Agregar las medias de x e y de ambos subgrupos a las listas
                    medias_x.extend([media_x1, media_x2])
                    medias_y.extend([media_y1, media_y2])
                elif min_y == 1 and max_y == 4 and 2 not in [y for _, y in coord]:
                    # Dividir el grupo en dos subgrupos
                    subgrupo1 = [p for p in coord if p[1] == 1]
                    subgrupo2 = [p for p in coord if p[1] == 4]

                    if (len(subgrupo2) == 2):
                        a = 0
                        b = 2
                    if (len(subgrupo2) == 4):
                        a = -2
                        b = -0.1
                    # Calcular la media de las coordenadas x e y para el primer subgrupo
                    media_x1 = sum(x for x, _ in subgrupo1) / len(subgrupo1)
                    media_y1 = sum(y for _, y in subgrupo1) / len(subgrupo1)

                    # Calcular la media de las coordenadas x e y para el segundo subgrupo
                    media_x2 = sum(x for x, _ in subgrupo2) / len(subgrupo2)
                    media_y2 = sum(y for _, y in subgrupo2) / len(subgrupo2)

                    # esto aplica para ambos subgrupos
                    max_x = max(x for x, _ in subgrupo2)
                    min_x = min(x for x, _ in subgrupo2)
                    width1 = max_x - min_x

                    if (len(subgrupo1) == 1):
                        height = 0.8
                        width1 = 0.8
                        a = 0
                        b = 0

                    # Dibujar un arco para el primer subgrupo   - x no varía en cada subgrupo
                    arc1 = Arc((media_x1, media_y1), width1 + a, height - b, angle=270, theta1=90, theta2=270,
                               edgecolor=edgecolor, lw=2)
                    ax.add_patch(arc1)

                    # Dibujar un arco para el segundo subgrupo
                    arc2 = Arc((media_x2, media_y2), width1 + a, height - b, angle=90, theta1=90, theta2=270,
                               edgecolor=edgecolor, lw=2)
                    ax.add_patch(arc2)

                    # Agregar las medias de x e y de ambos subgrupos a las listas
                    medias_x.extend([media_x1, media_x2])
                    medias_y.extend([media_y1, media_y2])
                else:
                    # Crear y dibujar la elipse con el borde de color aleatorio
                    ellipse = Ellipse(xy=(media_x, media_y), width=width, height=height, edgecolor=edgecolor, fc='none',
                                      lw=2)
                    ax.add_patch(ellipse)

        if grupos != None:
            # Llamar a la función para dibujar la elipse para cada grupo en resultado
            colores_usados = set()  # Inicializar un conjunto para mantener un seguimiento de los colores utilizados
            for grupo in resultado:
                graficar_elipse(grupo, colores_usados)
                print(grupo)

            # Crear la tabla
            tabla = ax.table(cellText=datos, loc='center', cellLoc='center')

            # Estilo de la tabla
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(12)
            tabla.scale(0.8, 3.7)  # Ajusta el tamaño de la tabla

            # Configurar los límites del eje
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 6)

            # Configurar el formato de los ticks en el eje x para mostrar enteros
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

            # Mostrar la figura
            plt.gca().set_aspect('equal', adjustable='box')  # Para mantener la relación de aspecto igual
            ax.axis('off')
    #if grupos != None:
        #plt.show()


#_____________________________________________________________________________________________________________________

# Función para mostrar la ventana de Instrucciones
def mostrar_instrucciones():
    instrucciones_window = tk.Toplevel(root)
    instrucciones_window.title("Instrucciones")
    instrucciones_window.configure(bg="black")

    instrucciones_label = tk.Label(instrucciones_window, text="Instrucciones:\n"
                                                            "1. Selecciona el número de variables.\n"
                                                            "2. Ingresa la función booleana en el formato deseado\n"
                                                            "3. Haz clic en el botón 'Generar Resultados' para ver las tablas.\n"
                                                            "4. ¡Disfruta usando la aplicación!\n\n"
                                                            "Recuerda:\n"
                                                            "- La negación (o NOT) se escribe con ' (Ejemplo: x')\n"
                                                            "- La unión (o AND) se escribe con ^ (Ejemplo: x^y)\n"
                                                            "- La conjunción (o OR) se escribe con v (Ejemplo: x v y)", padx=20, pady=20, bg="black", fg="white")
    instrucciones_label.pack()

# Función para mostrar la ventana de Créditos
def mostrar_creditos():
    creditos_window = tk.Toplevel(root)
    creditos_window.title("Créditos")
    creditos_window.configure(bg="black")

    creditos_label = tk.Label(creditos_window, text="Créditos:\n\n"
                                                    "Integrantes del grupo:\n"
                                                    "- Maria Fe Bojorquez Ancasi (U202313071) \n"
                                                    "- Marsi Valeria Figueroa Larragan (U202220990) \n"
                                                    "- Brenda Lucía Gamio Upiachihua (U202120344) \n"
                                                    "- Kael Valentino Lagos Rivera (U202210104) \n"
                                                    "- Liam Mikael Quino Neff (U20221E167) \n\n"
                                                    " Desarrolladores del código: \n"
                                                    "- Marsi Valeria Figueroa Larragan (U202220990) \n"
                                                    "- Liam Mikael Quino Neff (U20221E167) \n\n"
                                                    "Docente: Jonathan Abraham Sueros Zarate", padx=20, pady=20, bg="black", fg="white")
    creditos_label.pack()

# Función para salir de la aplicación
def salir_aplicacion():
    root.destroy()

# Crear la ventana principal
root = ctk.CTk()
root.title("Mapa de Karnaugh")

# Crear y configurar la interfaz gráfica
numVarLabel = ctk.CTkLabel(root, text="Nro de Variables")
numVarLabel.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

numVarVar = tk.IntVar(value=2)

twoRadioButton = ctk.CTkRadioButton(root, text="2", variable=numVarVar, value=2)
twoRadioButton.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

threeRadioButton = ctk.CTkRadioButton(root, text="3", variable=numVarVar, value=3)
threeRadioButton.grid(row=0, column=2, padx=20, pady=20, sticky="ew")

fourRadioButton = ctk.CTkRadioButton(root, text="4", variable=numVarVar, value=4)
fourRadioButton.grid(row=0, column=3, padx=20, pady=20, sticky="ew")

boolLabel = ctk.CTkLabel(root, text="Función booleana")
boolLabel.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

boolEntry = ctk.CTkEntry(root, placeholder_text="(x'^y)v(y'^z)")
boolEntry.grid(row=1, column=1, columnspan=3, padx=20, pady=20, sticky="ew")

generateResultsButton = ctk.CTkButton(root, text="Generar Resultados", command=generateResults)
generateResultsButton.grid(row=2, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

instruccionesButton = ctk.CTkButton(root, text="Instrucciones", command=mostrar_instrucciones)
instruccionesButton.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

creditosButton = ctk.CTkButton(root, text="Créditos", command=mostrar_creditos)
creditosButton.grid(row=3, column=2, padx=20, pady=20, sticky="ew")

salirButton = ctk.CTkButton(root, text="Salir", command=salir_aplicacion)
salirButton.grid(row=3, column=3, padx=20, pady=20, sticky="ew")

# Ejecutar la aplicación
root.mainloop()

#Ejemplos tradicionales:
#(x'^y)v(x^y)
#(x'^y'^z')v(x'^y'^z)v(x'^y^z)v(x^y'^z)
#(w'^x^y)v(w'^x^y')v(x'^y^z)v(x^y^z)

#Ejemplos esquinas:
#(x'^y'^z'^w')v(x'^y'^z^w')v(x^y'^z'^w')v(x^y'^z^w')
#(w)v(z)v(y)v(x)
